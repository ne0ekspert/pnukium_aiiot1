#include <Servo.h>

int v = 90;
Servo servo;

void setup() {
    servo.attach(7);

    servo.write(90);
    delay(1000);
    servo.write(30);
    delay(1000);
    servo.write(150);
    delay(1000);
    servo.write(90);

    Serial.begin(115200);
}

void loop() {
    bool skip = false;

    while (Serial.available()) {
        char c = Serial.read();

        switch(c) {
            case 'R':
                servo.write(150);
                break;
            case 'G':
                servo.write(30);
                break;
            default:
                skip = true;
                break;
        }

        delay(500);

        servo.write(90);
    }
}