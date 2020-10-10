#include <RH_ASK.h>
#include <SPI.h> // Not actually used but needed to compile
#include <M5StickC.h>
#include <stdio.h>
#include <stdlib.h>

RH_ASK driver;

void setup()
{
  M5.begin();
  M5.Lcd.setTextSize(3);
    Serial.begin(9600);    // Debugging only
    if (!driver.init())
         Serial.println("init failed");
}

void loop()
{   
    char data[1];
    
    for (char i = 32; i < 128; i++)
    {
    
    data[0] = i ;
    driver.send((uint8_t *)data, strlen(data));
    driver.waitPacketSent();
    M5.Lcd.setCursor(0,0);
    M5.Lcd.print(i);
    delay(100);
    M5.Lcd.fillScreen(BLACK);
    }
    
    M5.update();    
}
