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

void sendChars(int delayMilis) {
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.print("Transmiting...");
  
  for (char i = 32; i < 128 && !M5.BtnA.read(); i++) {
    data[0] = i ;
    driver.send((uint8_t *)data, strlen(data));
    driver.waitPacketSent();
    // M5.Lcd.fillScreen(BLACK);
    // M5.Lcd.setCursor(0, 0);
    // M5.Lcd.print(i);
    delay(delayMilis);
  }
}

void loop() {
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.print("Press A to begin");
  
  while(!M5.BtnA.read()){ delay(150); }
  while(M5.BtnA.read()){ delay(150); }

  while(!M5.BtnA.read()) { sendChars(10); }
  while(M5.BtnA.read()){ delay(150); }
  
  M5.update();
}
