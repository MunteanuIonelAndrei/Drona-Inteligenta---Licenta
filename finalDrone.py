import threading
import time
import math
import cv2
import tkinter as tk
import mediapipe as mp
from picamera2 import Picamera2
from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative
from ultralytics import YOLO
from collections import Counter

# Initializam camera
piCam = Picamera2()
cameraConfig = piCam.create_preview_configuration(main={"size": (860, 640), "format": "RGB888"})
piCam.configure(cameraConfig)
piCam.start()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDrawing = mp.solutions.drawing_utils

# Incarcam modelul
model = YOLO('/home/andre/Documents/env/best_ncnn_model')  

# Ne conectam la drona
print("Connecting to vehicle on: '/dev/drona'")
vehicle = connect('/dev/drona', wait_ready=True)

frameLock = threading.Lock()
latestFrame = None
processedFrame = None
commandQueue = []
threads = []
channelValues = {}
lastPwmValue = 600  
homeLocation = None  
yoloLandingInitiated = False  
yoloLandingFailed = False  
foundXCount = 0  

tip = [8, 12, 16, 20]
tipName = [8, 12, 16, 20]
fingers = []
finger = []
h, w = 0, 0

def isInside(point):
    (x, y) = point
    return 0 <= x <= w and 0 <= y <= h

@vehicle.on_message('RC_CHANNELS')
def rcListener(self, name, message):
    global channelValues
    channelValues = {
        '1': message.chan1_raw,
        '2': message.chan2_raw,
        '3': message.chan3_raw,
        '4': message.chan4_raw,
        '5': message.chan5_raw,
        '6': message.chan6_raw,
        '7': message.chan7_raw,
        '8': message.chan8_raw
    }

def setServo(servoN, pwmValue):
    global lastPwmValue
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  
        183,  # MAV_CMD_DO_SET_SERVO
        0,  
        servoN,  
        pwmValue,  
        0, 0, 0, 0, 0  
    )
    vehicle.send_mavlink(msg)
    if servoN == 9:
        lastPwmValue = pwmValue 

def mapValue(value, inMin, inMax, outMin, outMax):
    return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

def setGuidedMode():
    if vehicle.mode.name != "GUIDED":
        print("Setting vehicle to GUIDED mode")
        vehicle.mode = VehicleMode("GUIDED")
        while not vehicle.mode.name == "GUIDED":
            print(" Waiting for mode change ...")
            time.sleep(1)
        print("Mode changed to GUIDED")

def printStatus():
    gps = vehicle.gps_0
    ekfStatus = vehicle.ekf_ok
    currentLocation = vehicle.location.global_frame
    relativeLocation = vehicle.location.global_relative_frame

    print("GPS Status")
    print(f"GPS Count: {gps.satellites_visible}")
    print(f"GPS Lock: {'3D Lock' if gps.fix_type == 3 else 'No Lock'}")
    print(f"HDOP: {gps.eph}")
    print(f"VDOP: {gps.epv}")
    print(f"Course Over Ground: {vehicle.heading}")
    print(f"EKF status: {'OK' if ekfStatus else 'Not OK'}")
    print(f"Location is: {currentLocation}")
    print(f"Location (Relative): Altitude: {relativeLocation.alt} meters")

def statusMonitor():
    while True:
        printStatus()
        time.sleep(5)

statusThread = threading.Thread(target=statusMonitor)
statusThread.daemon = True
statusThread.start()

def captureFrames():
    global latestFrame
    while True:
        frame = piCam.capture_array()
        frame = cv2.flip(frame, -1)  # Rotim camera, e montata invers
        with frameLock:
            latestFrame = frame.copy()  
        time.sleep(0.01)  

def sendNedVelocity(velocityX, velocityY, velocityZ, duration):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000110,
        0, 0, 0,
        velocityX, velocityY, velocityZ,
        0, 0, 0,
        0, 0
    )

    iterations = int(duration / 0.1)
    for _ in range(iterations):
        vehicle.send_mavlink(msg)
        time.sleep(0.01)

def hover():
    print("Hovering in place")
    sendNedVelocity(0, 0, 0, 1)

def moveRelativeToHeading(speed, direction):
    heading = math.radians(vehicle.heading)
    print(f"Moving {direction} ")

    if direction == "forward":
        northVelocity = speed * math.cos(heading)
        eastVelocity = speed * math.sin(heading)
    elif direction == "backward":
        northVelocity = -speed * math.cos(heading)
        eastVelocity = -speed * math.sin(heading)
    elif direction == "right":
        northVelocity = -speed * math.sin(heading)
        eastVelocity = speed * math.cos(heading)
    elif direction == "left":
        northVelocity = speed * math.sin(heading)
        eastVelocity = -speed * math.cos(heading)
    else:
        raise ValueError("Invalid direction. Choose from 'forward', 'backward', 'right', 'left'.")

    sendNedVelocity(northVelocity, eastVelocity, 0, 1)

def changeAltitude(change):
    currentAltitude = vehicle.location.global_relative_frame.alt
    newAltitude = currentAltitude + change
    print(f"Changing altitude to {newAltitude} meters.")
    targetLocation = LocationGlobalRelative(vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon, newAltitude)
    vehicle.simple_goto(targetLocation)

def rotateYaw(direction):
    angle = 5  
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        angle,
        0,
        direction,
        1, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    print(f"Rotating yaw {'clockwise' if direction == 1 else 'counterclockwise'} by {angle} degrees")

def setHomeLocation():
    global homeLocation
    currentLocation = vehicle.location.global_frame
    homeLocation = currentLocation
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_DO_SET_HOME,
        0,
        1,
        0, 0, 0,
        currentLocation.lat,
        currentLocation.lon,
        currentLocation.alt
    )
    vehicle.send_mavlink(msg)
    print(f"Home location set to: {currentLocation}")

def waitForEkfOk(timeout=30):
    startTime = time.time()
    while not vehicle.ekf_ok and time.time() - startTime < timeout:
        print("Waiting for EKF to be OK...")
        time.sleep(1)
    if vehicle.ekf_ok:
        print("EKF is OK")
    else:
        print("EKF not OK after timeout")

def armDrone():
    setGuidedMode()
    setHomeLocation()
    waitForEkfOk()
    print("Arming motors")
    vehicle.armed = True
    timeout = 10
    startTime = time.time()
    while not vehicle.armed and time.time() - startTime < timeout:
        print("Waiting for arming...")
        time.sleep(1)
    if vehicle.armed:
        print("Drone is armed")
    else:
        print("Failed to arm drone within the timeout period")

def takeoffDrone(altitude):
    print(f"Taking off to {altitude} meters")
    vehicle.simple_takeoff(altitude)
    while True:
        currentAltitude = vehicle.location.global_relative_frame.alt
        print(f"Current altitude: {currentAltitude} meters")
        if currentAltitude >= altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def check():
    if a:
        x = a[4][2]
        for id in range(0, 2):
            if a[id][2] < x:
                return False
        return True

def findPosition(frame1, h, w):
    results = hands.process(frame1)
    handLandmarksList = []
    handednessList = []

    if results.multi_hand_landmarks:
        for idx, handLandmarks in enumerate(results.multi_hand_landmarks):
            mpDrawing.draw_landmarks(frame1, handLandmarks, mpHands.HAND_CONNECTIONS)
            handLandmarkPositions = []
            for id, lm in enumerate(handLandmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                handLandmarkPositions.append([id, cx, cy])
            handLandmarksList.append(handLandmarkPositions)
            handednessList.append(results.multi_handedness[idx].classification[0].label)

    return handLandmarksList, handednessList

def land():
    print("Initiating landing...")
    vehicle.mode = VehicleMode("LAND")

    setServo(9, 1650)


def handControlLanding():
    global h, w, finger, fingers, tip, processedFrame
    up = 0
    a = []
    while True:
        up = 0
        with frameLock:
            if latestFrame is not None:
                frameToProcess = latestFrame.copy()
                h, w, _ = frameToProcess.shape
                center = (int(w / 2), int(h / 2))
                handsPositions, handedness = findPosition(frameToProcess, h, w)

                for i, a in enumerate(handsPositions):
                    if len(a) != 0:
                        # Cum camera e invers, invartim si pozitia mainilor
                        if handedness[i] == 'Right':
                            handedness[i] = 'Left'
                        else:
                            handedness[i] = 'Right'

                        fingers = []
                        if handedness[i] == 'Right':
                            if a[4][1] > a[3][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                            for id in range(0, 4):
                                if a[tip[id]][2] < a[tip[id] - 2][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)
                        else:  # Mana stanga
                            if a[4][1] < a[3][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                            for id in range(0, 4):
                                if a[tip[id]][2] < a[tip[id] - 2][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)

                        x = fingers
                        c = Counter(x)
                        up = c[1]
                        down = 5 - up
                        print(f'This many fingers are up for {handedness[i]} hand - ', up)

                if up == 5:
                    moveByHand(a, center, frameToProcess)
                elif up == 2 and handedness[i] == 'Right':
                    print("Moving forward")
                    moveRelativeToHeading(0.2, "forward")
                elif up == 2 and handedness[i] == 'Left':
                    print("Moving backward")
                    moveRelativeToHeading(0.2, "backward")
                elif up == 1 and handedness[i] == 'Right' and check:
                    print("Landing command received via hand signal")
                    processedFrame = None
                    vehicle.mode = VehicleMode("LAND")
                    break

                processedFrame = frameToProcess.copy()

                if len(a) == 0:
                    stopMovement()
        time.sleep(0.01)  

def moveByHand(a, center, frame):
    if a and (isInside((a[0][1], a[0][2])) or isInside((a[5][1], a[5][2])) or isInside((a[17][1], a[17][2]))):
        middleX = int((a[0][1] + a[5][1] + a[17][1]) / 3)
        middleY = int((a[0][2] + a[5][2] + a[17][2]) / 3)
        middle = (middleX, middleY)
        cv2.rectangle(frame, (middleX - 40, middleY - 30), (middleX + 40, middleY + 30), (0, 255, 0), 2)
        if middleX - 40 < center[0] < middleX + 40 and middleY - 30 < center[1] < middleY + 30:
            cv2.circle(frame, (middleX, middleY), 5, (0, 255, 0), -1)
            stopMovement()
        else:
            cv2.circle(frame, (middleX, middleY), 5, (0, 0, 255), -1)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.line(frame, center, middle, (0, 0, 0), 3)
            if middleX - 40 > center[0]:
                print("Moving right")
                moveRelativeToHeading(0.2, "right")
            elif middleX + 40 < center[0]:
                print("Moving left")
                moveRelativeToHeading(0.2, "left")

def returnToLaunch():
    print("Returning to Launch")
    vehicle.mode = VehicleMode("RTL")

def stopMovement():
    print("Stopping movement")
    hover()

def killMotors():
    print("Killing motors")
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)

def rebootPixhawk():
    print("Rebooting Pixhawk")
    vehicle._master.mav.command_long_send(
        vehicle._master.target_system,
        vehicle._master.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        0,
        1,
        0, 0, 0, 0, 0, 0, 0
    )

def checkRcInput():
    lastRcValue = None
    threshold = 6
    while True:
        rcValue = channelValues.get('5')
        if rcValue is not None:
            if lastRcValue is not None and abs(rcValue - lastRcValue) > threshold:
                pwmValue = mapValue(rcValue, 1000, 2000, 620, 1650)
                setServo(9, pwmValue)
                print(f"RC Value: {rcValue}, PWM Value: {pwmValue}")
            lastRcValue = rcValue
        time.sleep(0.1)

def displayVideo():
    global latestFrame, processedFrame
    while True:
        with frameLock:
            frameToDisplay = processedFrame if processedFrame is not None else latestFrame
            if frameToDisplay is None:
                continue

        cv2.imshow("Video Feed", frameToDisplay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def isWithinInnerRectangle(point, rectangle):
    x, y = point
    xMin, yMin, xMax, yMax = rectangle
    return xMin <= x <= xMax and yMin <= y <= yMax

def startYoloLanding():
    global latestFrame, processedFrame, yoloLandingInitiated, yoloLandingFailed, foundXCount

    foundXCount = 0
    setServo(9, 1650)

    targetAltitude = vehicle.location.global_relative_frame.alt
    minAltitude = 2.0
    stepDescent = 1.0
    detectionTimeout = 3

    def getInnerRectangle(frameWidth, frameHeight):
        return (
            int(frameWidth * 0.3),
            int(frameHeight * 0.3),
            int(frameWidth * 0.7),
            int(frameHeight * 0.7)
        )

    innerRect = None

    def descentStep():
        nonlocal targetAltitude
        targetAltitude -= stepDescent
        print(f"Descending to {targetAltitude} meters")
        changeAltitude(-stepDescent)

    def unloadPackage():
        print("Unloading package")
        setServo(10, 1300)
        time.sleep(5)  
        setServo(9, 620)

    while targetAltitude > minAltitude:
        startTime = time.time()
        foundX = False

        while time.time() - startTime < detectionTimeout:
            with frameLock:
                if latestFrame is not None:
                    frameToProcess = latestFrame.copy()

            if frameToProcess is not None:
                h, w, _ = frameToProcess.shape
                center = (int(w / 2), int(h / 2))

                if innerRect is None:
                    innerRect = getInnerRectangle(w, h)

                cv2.rectangle(frameToProcess, (innerRect[0], innerRect[1]), (innerRect[2], innerRect[3]), (255, 0, 0), 2)

                results = model.predict(frameToProcess, show=False, conf=0.33)
                if results:
                    for result in results:
                        if result.boxes:
                            box = result.boxes[0]
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            cv2.rectangle(frameToProcess, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{box.conf.item():.2f}"
                            cv2.putText(frameToProcess, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                            rectCenterX = (x1 + x2) // 2
                            rectCenterY = (y1 + y2) // 2
                            cv2.circle(frameToProcess, (rectCenterX, rectCenterY), 5, (0, 255, 0), -1)

                            foundX = True

                            if targetAltitude < 6:
                                foundXCount += 1

                            if isWithinInnerRectangle((rectCenterX, rectCenterY), innerRect):
                                if abs(rectCenterX - center[0]) > 40:
                                    if rectCenterX < center[0]:
                                        moveRelativeToHeading(0.1, "left")
                                        print("Moving left")
                                    elif rectCenterX > center[0]:
                                        moveRelativeToHeading(0.1, "right")
                                        print("Moving right")
                                elif abs(rectCenterY - center[1]) > 30:
                                    if rectCenterY > center[1]:
                                        moveRelativeToHeading(0.1, "backward")
                                        print("Moving backward")
                                    elif rectCenterY < center[1]:
                                        moveRelativeToHeading(0.1, "forward")
                                        print("Moving forward")
                                else:
                                    if targetAltitude <= 3:
                                        if foundXCount > 3:
                                            print("Centered over the X shape at 2.5 meters or below, unloading the package")
                                            threading.Thread(target=unloadPackage).start()
                                            hover()
                                            yoloLandingInitiated = True
                                            return
                                    else:
                                        print("Centered over the X shape, descending 1 meter")
                                        threading.Thread(target=descentStep).start()
                                        break
                            else:
                                if rectCenterX < innerRect[0]:
                                    if rectCenterY < innerRect[1]:
                                        moveRelativeToHeading(0.2, "left")
                                        moveRelativeToHeading(0.2, "forward")
                                        print("Moving left and forward")
                                    elif rectCenterY > innerRect[3]:
                                        moveRelativeToHeading(0.2, "left")
                                        moveRelativeToHeading(0.2, "backward")
                                        print("Moving left and backward")
                                    else:
                                        moveRelativeToHeading(0.2, "left")
                                        print("Moving left")
                                elif rectCenterX > innerRect[2]:
                                    if rectCenterY < innerRect[1]:
                                        moveRelativeToHeading(0.2, "right")
                                        moveRelativeToHeading(0.2, "forward")
                                        print("Moving right and forward")
                                    elif rectCenterY > innerRect[3]:
                                        moveRelativeToHeading(0.2, "right")
                                        moveRelativeToHeading(0.2, "backward")
                                        print("Moving right and backward")
                                    else:
                                        moveRelativeToHeading(0.2, "right")
                                        print("Moving right")
                                elif rectCenterY < innerRect[1]:
                                    moveRelativeToHeading(0.2, "forward")
                                    print("Moving forward")
                                elif rectCenterY > innerRect[3]:
                                    moveRelativeToHeading(0.2, "backward")
                                    print("Moving backward")

            processedFrame = frameToProcess.copy()
            time.sleep(0.01)

        if not foundX:
            threading.Thread(target=lambda: changeAltitude(-0.5)).start()
            time.sleep(0.5)  
    if targetAltitude <= 2.5 and foundXCount > 3:
        print("Centered over the X shape at 2.5 meters or below, unloading the package")
        threading.Thread(target=unloadPackage).start()
        hover()
        yoloLandingInitiated = True
    else:
        print("Reached minimum altitude of 2 meters, hovering")
        hover()
        setServo(9, 620)
        yoloLandingFailed = True

    processedFrame = None

class DroneControlUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Drone Control Interface")
        self.controlsFrame = tk.Frame(master)
        self.controlsFrame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        for i in range(10): 
            self.controlsFrame.grid_rowconfigure(i, weight=1)
        for i in range(6):
            self.controlsFrame.grid_columnconfigure(i, weight=1)

        self.createButtons()
        self.createTextFields()

    def createButtons(self):
        tk.Button(self.controlsFrame, text="Up", command=lambda: changeAltitude(0.2)).grid(row=0, column=1, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Down", command=lambda: changeAltitude(-0.2)).grid(row=2, column=1, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Yaw Left", command=lambda: rotateYaw(-1)).grid(row=1, column=0, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Yaw Right", command=lambda: rotateYaw(1)).grid(row=1, column=2, pady=5, padx=5, sticky='nsew')
        
        tk.Button(self.controlsFrame, text="Forward", command=lambda: moveRelativeToHeading(0.2, "forward")).grid(row=0, column=4, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Backward", command=lambda: moveRelativeToHeading(0.2, "backward")).grid(row=2, column=4, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Left", command=lambda: moveRelativeToHeading(0.2, "left")).grid(row=1, column=3, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Right", command=lambda: moveRelativeToHeading(0.2, "right")).grid(row=1, column=5, pady=5, padx=5, sticky='nsew')

        tk.Button(self.controlsFrame, text="Arm", command=armDrone).grid(row=3, column=0, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Takeoff", command=lambda: takeoffDrone(2.5)).grid(row=3, column=2, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="RTL", command=returnToLaunch).grid(row=4, column=2, columnspan=2, pady=5, padx=5, sticky='nsew')

        tk.Button(self.controlsFrame, text="Land", command=land).grid(row=4, column=0, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Kill Motors", command=killMotors).grid(row=5, column=0, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Reboot", command=rebootPixhawk).grid(row=5, column=2, columnspan=2, pady=5, padx=5, sticky='nsew')

        tk.Button(self.controlsFrame, text="Hand Control Land", command=lambda: startThread(handControlLanding)).grid(row=6, column=0, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="YOLO Land", command=lambda: startThread(startYoloLanding)).grid(row=6, column=2, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Tilt Camera Up", command=self.tiltCameraUp).grid(row=7, column=0, columnspan=2, pady=5, padx=5, sticky='nsew')

        tk.Button(self.controlsFrame, text="Tilt Camera Down", command=self.tiltCameraDown).grid(row=7, column=2, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Load", command=lambda: setServo(10, 2500)).grid(row=8, column=0, columnspan=2, pady=5, padx=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Unload", command=lambda: setServo(10, 1300)).grid(row=8, column=2, columnspan=2, pady=5, padx=5, sticky='nsew')



    def createTextFields(self):
        tk.Label(self.controlsFrame, text="Latitude:").grid(row=3, column=4, pady=(5, 2), sticky='e')
        self.latitudeEntry = tk.Entry(self.controlsFrame, width=20)
        self.latitudeEntry.grid(row=3, column=5, pady=(5, 2), sticky='w')

        tk.Label(self.controlsFrame, text="Longitude:").grid(row=4, column=4, pady=2, sticky='e')
        self.longitudeEntry = tk.Entry(self.controlsFrame, width=20)
        self.longitudeEntry.grid(row=4, column=5, pady=2, sticky='w')

        tk.Label(self.controlsFrame, text="Altitude:").grid(row=5, column=4, pady=2, sticky='e')
        self.altitudeEntry = tk.Entry(self.controlsFrame, width=20)
        self.altitudeEntry.grid(row=5, column=5, pady=2, sticky='w')

        tk.Button(self.controlsFrame, text="SimpleGoTo", command=self.simpleGoto, width=15).grid(row=6, column=4, columnspan=2, pady=5, sticky='nsew')
        tk.Button(self.controlsFrame, text="Start Mission", command=lambda: startThread(self.startMission), width=15).grid(row=7, column=4, columnspan=2, pady=5, sticky='nsew')

    def tiltCameraUp(self):
        global lastPwmValue
        if lastPwmValue > 620:
            newPwmValue = max(lastPwmValue - 50, 620)  
            setServo(9, newPwmValue)

    def tiltCameraDown(self):
        global lastPwmValue
        if lastPwmValue < 1650:
            newPwmValue = min(lastPwmValue + 50, 1650)  
            setServo(9, newPwmValue)
            
    def haversineDistance(self, lat1, lon1, lat2, lon2):
        R = 6371  
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def simpleGoto(self):
        try:
            latitudeInput = self.latitudeEntry.get()
            longitudeInput = self.longitudeEntry.get()
            altitude = float(self.altitudeEntry.get())  

            # Locatia curenta
            currentLocation = vehicle.location.global_relative_frame
            currentLat = currentLocation.lat
            currentLon = currentLocation.lon

            # Folosim locatia curenta daca aceasta nu se precizeaza de catre operator
            if latitudeInput.strip() == "":
                latitude = currentLat
            else:
                latitude = float(latitudeInput)

            if longitudeInput.strip() == "":
                longitude = currentLon
            else:
                longitude = float(longitudeInput)

            distanceToTarget = self.haversineDistance(currentLat, currentLon, latitude, longitude)

            if distanceToTarget <= 1.0:
                print(f"Starting mission to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude} meters (relative)")
                targetLocation = LocationGlobalRelative(latitude, longitude, altitude)
                vehicle.simple_goto(targetLocation)
            else:
                print(f"Cannot start mission: Target location is {distanceToTarget:.2f} km away, which is more than the allowed 1 km distance.")
        except ValueError:
            print("Invalid input: Please enter valid numeric values for latitude, longitude, and altitude.")

    def startMission(self):
        global yoloLandingInitiated, yoloLandingFailed
        try:
            latitudeInput = self.latitudeEntry.get()
            longitudeInput = self.longitudeEntry.get()
            altitude = float(self.altitudeEntry.get())  

            currentLocation = vehicle.location.global_relative_frame
            currentLat = currentLocation.lat
            currentLon = currentLocation.lon

            if latitudeInput.strip() == "":
                latitude = currentLat
            else:
                latitude = float(latitudeInput)

            if longitudeInput.strip() == "":
                longitude = currentLon
            else:
                longitude = float(longitudeInput)

            distanceToTarget = self.haversineDistance(currentLat, currentLon, latitude, longitude)

            if distanceToTarget <= 1.0:
                if not vehicle.armed or vehicle.location.global_relative_frame.alt < 1.5:
                    print("Drone is not armed or not airborne. Arming and taking off to 2 meters.")
                    armDrone()
                    takeoffDrone(2.0)

                print(f"Starting mission to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude} meters (relative)")
                targetLocation = LocationGlobalRelative(latitude, longitude, altitude)
                vehicle.simple_goto(targetLocation)

                while True:
                    if yoloLandingFailed:
                        print("YOLO landing failed. Stopping mission.")
                        return
                    currentLocation = vehicle.location.global_relative_frame
                    distanceToTarget = self.haversineDistance(currentLocation.lat, currentLocation.lon, latitude, longitude)
                    print(f"Distance to target: {distanceToTarget:.2f} km")
                    if distanceToTarget <= 0.005:  
                        print("Target location reached. Starting YOLO landing.")
                        startThread(startYoloLanding)
                        break
                    time.sleep(2)
                # Continuam misiunea daca s-a livrat pachetul
                while not yoloLandingInitiated:
                    if yoloLandingFailed:
                        print("YOLO landing failed. Stopping mission.")
                        return
                    time.sleep(1)

                print("Returning to home location.")
                vehicle.simple_goto(LocationGlobalRelative(homeLocation.lat, homeLocation.lon, altitude))

                while True:
                    if yoloLandingFailed:
                        print("YOLO landing failed. Stopping mission.")
                        return
                    currentLocation = vehicle.location.global_relative_frame
                    distanceToHome = self.haversineDistance(currentLocation.lat, currentLocation.lon, homeLocation.lat, homeLocation.lon)
                    print(f"Distance to home: {distanceToHome:.2f} km")
                    if distanceToHome <= 0.005:  
                        print("Home location reached. Descending to 2.5 meters.")
                        vehicle.simple_goto(LocationGlobalRelative(homeLocation.lat, homeLocation.lon, 2.5))
                        while vehicle.location.global_relative_frame.alt > 2.6:
                            time.sleep(1)
                        print("Starting hand control landing.")
                        startThread(handControlLanding)
                        break
                    time.sleep(2)

            else:
                print(f"Cannot start mission: Target location is {distanceToTarget:.2f} km away, which is more than the allowed 1 km distance.")
        except ValueError:
            print("Invalid input: Please enter valid numeric values for latitude, longitude, and altitude.")

def startThread(target):
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    threads.append(thread)

startThread(statusMonitor)
startThread(checkRcInput)

if __name__ == "__main__":
    setServo(9, 620)  
    startThread(captureFrames)
    startThread(displayVideo)

    root = tk.Tk()
    app = DroneControlUI(root)
    root.mainloop()
    vehicle.mode = VehicleMode("LAND")
    vehicle.close()
    print("Completed")


