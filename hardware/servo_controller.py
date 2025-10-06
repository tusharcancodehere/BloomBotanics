#!/usr/bin/env python3
"""
Servo Controller for Rotating Scarecrow / Camera Cam
BloomBotanics - October 2025
"""

import RPi.GPIO as GPIO
import time

class ServoController:
    """Servo motor control over PWM"""
    def __init__(self, pin=17):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)  # 50Hz PWM
        self.pwm.start(0)
        print(f"Servo initialized on GPIO {self.pin}")

    def set_angle(self, angle):
        """Set servo angle (0-180 degrees)"""
        duty_cycle = 2 + (angle / 18)
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(0)

    def rotate_fro(self, start_angle=0, end_angle=180, step=10, delay=0.2):
        """Back and forth rotation over 180 degrees"""
        try:
            while True:
                # 0 to 180 degrees
                for angle in range(start_angle, end_angle + 1, step):
                    self.set_angle(angle)
                    time.sleep(delay)
                # 180 to 0 degrees
                for angle in range(end_angle, start_angle - 1, -step):
                    self.set_angle(angle)
                    time.sleep(delay)
        except KeyboardInterrupt:
            print("Servo rotation stopped by user.")

    def cleanup(self):
        """Cleanup GPIO resources"""
        self.pwm.stop()
        GPIO.cleanup(self.pin)
        print("Servo cleanup complete")
