#include <RH_ASK.h>
#include <SPI.h> // Not actually used but needed to compile
#include <M5StickC.h>

RH_ASK driver;
char data[1];

void setup() {
  M5.begin();
  M5.Lcd.setTextSize(2);
  if (!driver.init()) {
    M5.Lcd.print("driver init failure");
    while (1) {}
  }
}

void sendByteAndWait(int delayMilis) {
    data[0] = random(0, 256);
    driver.send((uint8_t *)data, strlen(data));
    driver.waitPacketSent();
    delay(delayMilis);
}

void loop() {
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setCursor(0, 50);
  M5.Lcd.print("Press A to begin");
  while(!M5.BtnA.read()){ delay(150); }
  while(M5.BtnA.read()){ delay(150); }
  
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setCursor(0, 50);
  M5.Lcd.print("Transmiting");
  while(!M5.BtnA.read()) { sendByteAndWait(10); }
  while(M5.BtnA.read()){ delay(150); }
  
  M5.update();
}
