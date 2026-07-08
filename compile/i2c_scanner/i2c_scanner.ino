// Standalone I2C bus scanner for XIAO ESP32-S3.
// Flash this alone (separate from compile.ino) to confirm both the
// MPU-6050 (0x68) and MLX90640 (0x33) respond on the shared I2C bus
// before wiring changes are made in the main sketch.
//
// Default XIAO ESP32-S3 I2C pins: SDA=D4(GPIO5), SCL=D5(GPIO6)
// matching IMU_I2C_SDA / IMU_I2C_SCL in compile.ino.

#include <Wire.h>

#define SDA_PIN 5  // D4
#define SCL_PIN 6  // D5

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  delay(500);

  Serial.println();
  Serial.printf("I2C scanner starting on SDA=GPIO%d SCL=GPIO%d\n", SDA_PIN, SCL_PIN);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
}

void loop() {
  Serial.println("Scanning...");
  int found = 0;
  bool sawMPU = false;
  bool sawMLX = false;

  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();

    if (err == 0) {
      Serial.printf("  Found device at 0x%02X", addr);
      if (addr == 0x68) {
        Serial.print("  <-- MPU-6050 (IMU)");
        sawMPU = true;
      } else if (addr == 0x33) {
        Serial.print("  <-- MLX90640 (thermal)");
        sawMLX = true;
      }
      Serial.println();
      found++;
    }
  }

  if (found == 0) {
    Serial.println("  No I2C devices found. Check wiring/pull-ups/power.");
  } else {
    Serial.printf("Done. %d device(s) found.\n", found);
    Serial.printf("  MPU-6050 (0x68): %s\n", sawMPU ? "OK" : "MISSING");
    Serial.printf("  MLX90640 (0x33): %s\n", sawMLX ? "OK" : "MISSING");
  }

  Serial.println();
  delay(2000);
}
