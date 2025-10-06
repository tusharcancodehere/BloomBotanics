# 🔌 BloomBotanics Wiring Guide

Complete wiring diagram and connections for Raspberry Pi 4 agricultural monitoring system.

## 📋 Components List

### Core Components
- **Raspberry Pi 4** (1GB+ RAM recommended)
- **DHT22** Temperature/Humidity Sensor
- **Soil Moisture Sensor** (with analog output)
- **Rain Sensor Module** (with digital output)
- **Pi Camera Module** (V2 or later)
- **16x2 LCD Display** with I2C backpack
- **Single Channel Relay Module** (5V/10A)
- **SIM800L GSM Module**
- **12V PWM Fan** (for cooling)
- **MCP3008 ADC** (for analog sensors)

### Supporting Components
- **Breadboard** or **Perfboard**
- **Jumper Wires** (Male-Male, Male-Female)
- **10kΩ Resistors** (pull-up for DHT22)
- **Power Supply** (5V 3A for Pi + 12V for pump/fan)
- **MicroSD Card** (16GB+ Class 10)

## 🔧 GPIO Pin Assignments

| Component | Connection | GPIO Pin | Physical Pin | Notes |
|-----------|------------|----------|--------------|-------|
| **DHT22** | Data | GPIO 4 | Pin 7 | Requires 10kΩ pull-up |
| **Soil Sensor** | Digital | GPIO 3 | Pin 5 | Via MCP3008 ADC |
| **Rain Sensor** | Digital | GPIO 27 | Pin 13 | Built-in pull-up |
| **Rain Sensor** | Analog | GPIO 22 | Pin 15 | Via MCP3008 ADC |
| **Relay Module** | Control | GPIO 18 | Pin 12 | 5V logic level |
| **Fan Controller** | PWM | GPIO 14 | Pin 8 | 25kHz PWM |
| **LCD Display** | SDA | GPIO 2 | Pin 3 | I2C Address: 0x27 |
| **LCD Display** | SCL | GPIO 3 | Pin 5 | I2C Clock |
| **SIM800L** | TX | GPIO 14 | Pin 8 | UART Serial |
| **SIM800L** | RX | GPIO 15 | Pin 10 | UART Serial |

## 📐 Detailed Wiring Diagrams

### DHT22 Temperature/Humidity Sensor
