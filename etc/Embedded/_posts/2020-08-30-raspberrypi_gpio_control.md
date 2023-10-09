---
layout: post
title: Raspberry Pi GPIO Control
subtitle: Survo Moter Control With Raspberry Pi
keyword: '[RaspberryPi]'
category_num : 2
---

# Raspberry Pi GPIO Control

- Raspberry 4 with Ubuntu 20.04.1 LTS
- [Raspberry Pi Document](<https://www.raspberrypi.org/documentation/>)
- [raspberry-gpio-python](<https://sourceforge.net/p/raspberry-gpio-python/wiki/Examples/>)
- update at 2020.08.30

## Raspberry Pi GPIO

라즈베리파이를 보게 되면 아래 사진과 같이 40개의 핀(라즈베리파이 4 기준)이 있는 것을 확인할 수 있는데 이를 **[GPIO](<https://www.raspberrypi.org/documentation/usage/gpio/>)(General Purpose Input/Output) Pin**이라고 한다. 각각의 Pin은 고유의 목적을 가지고 있으며 사용자는 필요에 따라 각 Pin으로 데이터를 전송하거나 전송 받을 수 있다(그림출처 - RaspberryPi 홈페이지).

<img src="{{site.image_url}}/development/GPIO_pin_diagram.png" style="width:35em; display: block; margin: 0em auto">

참고로 라즈베리파이의 입출력 정보는 `$ pinout` 명령어를 통해 아래와 같이 확인할 수 있다.

<img src="{{site.image_url}}/development/pinout_screen.png" style="width:35em; display: block; margin: 0em auto">

Pin의 종류는 다음과 같이 크게 세 가지로 나눠볼 수 있다.

- Power: 전류를 전달하는 Pin, 5V와 3V3으로 나눠진다.
- Ground: 전류를 전달받는 Pin, GND로 표기되어 있다.
- Control: 시그널을 보내 통제하는 데에 사용되는 Pin

참고로 GND는 모든 Pin과 연결되어 있어 어떠한 Pin과도 매칭이 될 수 있다고 한다([pinout](<https://pinout.xyz/pinout/ground>)).

### GPIO with PYTHON

파이썬 코드로 GPIO를 사용하기 위해서는 `RPi.GPIO`라는 파이썬 패키지를 다음 명령어로 설치해주어야 한다. 패키지 사용과 관련된 정보는 [raspberry-gpio-python](<https://sourceforge.net/p/raspberry-gpio-python/wiki/Examples/>)에서 확인할 수 있다.

```
$ pip install RPi.GPIO
```

Python 코드로 GPIO를 사용하기 위해 가장 먼저 해주어야 하는 것은 Numbering system을 설정하고(board/bcm) 사용하고자 하는 Pin 번호를 설정해주는 것이다. 

```python
import RPi.GPIO as GPIO


servoPIN = 17

# set numbering system
GPIO.setmode(GPIO.BCM) # or GPIO.setmode(GPIO.BOARD)

# pin numbering
GPIO.setup(servoPIN, GPIO.OUT, initial=GPIO.HIGH)
```

#### Numbering System

Numbering System이란 40개의 Pin을 어떤 방법으로 이름을 붙일지에 관한 것이라고 할 수 있다. 위의 코드에서 확인할 수 있듯 
**GPIO.BOARD**와 **GPIO.BCM** 두 가지 방법이 존재하는데 GPIO.BOARD는 40개 Pin 각각에 순서대로 부여된 숫자(위의 pinout 그림에서 괄호()안의 숫자)로 Pin의 이름을 붙이는 방식을 지칭하고, GPIO.BCM는 GPIO 번호(위의 pinout 그림에서 GPIO 바로 뒤에 오는 숫자)로 Pin의 이름을 붙이는 방식이다. 위의 코드에서는 `GPIO.BCM`으로 설정하고 있으므로, 라즈베리파이의 11번 Pin을 사용하겠다는 것을 의미한다.

#### Pin Numbering

Numbering System을 적용한 뒤에는 Numbering System에 맞게 사용할 Pin의 숫자를 전달하여 Output Pin을 설정해주어야 한다.

## Servo Motor

[위키](<https://ko.wikipedia.org/wiki/%EC%84%9C%EB%B3%B4_%EA%B8%B0%EA%B5%AC>)에 따르면 서보 모터란 물체의 위치·방위·자세·회전 속도 등을 제어량으로 하고 목표치의 변화에 뒤따르도록 구성된 자동제어계로 정의된다. 쉽게 말해 정해진 값에 따라 움직이는 모터를 말한다. 포스팅에 사용한 서보 모터는 Futaba S3003이다. 대부분의 서보모터와 마찬가지로 Futaba S3003은 세 개의 와이어를 가지며, 각각의 목적과 실험을 위해 연결한 라즈베리파이 핀 번호는 다음과 같다.

|Color|Purpose|Board Pin Num|
|:----:|:---:|:---:|
|Black(Brown)|Ground, Negative Terminal| 2(5V) |
|Red|Servo Power, Positive Terminal| 14(GND) |
|White(Yellow)|Servo Control Signal| 11(GPIO17) |

### GPIO with Python

서보 모터를 컨트롤 하는데 사용하는 Python 코드는 다음과 같다. 코드는 다음 [홈페이지](<https://sourceforge.net/p/raspberry-gpio-python/wiki/PWM/>)를 참고했다.

```python
import RPi.GPIO as GPIO
import time

# 1. Pin Number Setttings
servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT, initial=GPIO.HIGH)

# 2. PWM Settings
p = GPIO.PWM(servoPIN, 50)
p.start(0)

# 3. Control Signal
try:
  while True:
    p.ChangeDutyCycle(5)
    time.sleep(0.5)
    p.ChangeDutyCycle(10)
    time.sleep(0.5)
except KeyboardInterrupt:
  p.stop()

# 4. Clean Up
GPIO.cleanup()
```

각각이 의미하는 바는 다음과 같다.

#### PWM Settings: Frequency

**[PWM](<https://ko.wikipedia.org/wiki/%ED%8E%84%EC%8A%A4_%ED%8F%AD_%EB%B3%80%EC%A1%B0>)**(Pulse Width Modulation or PDM, Pulse Duration Modulation)이란 Pulse의 폭을 제어하여 임의의 정해진 출력 파형을 유도하는 방법이다. PWM은 기본적으로 한 cycle 내에서 High Voltage의 비율(Duty Cycle)을 설정하는 방식으로 동작하며 이를 통해 LED의 밝기, 모터의 회전 속도 및 횟수를 제어할 수 있게 된다.

```
p = GPIO.PWM(servoPIN, 50)
```

위 코드에서 50이란 servoPIN의 주파수를 50Hz로 설정하겠다는 것을 의미한다(기본적인 서보모터는 50Hz를 사용하여 제어한다). 즉 1초에 50사이클을 돌 수 있도록 설정하겠다는 것이다. 참고로 다음 코드로 실행 도중에 주파수를 25Hz로 변경할 수 있다.

```
p.ChangeFrequency(25) 
```

#### PWM Settings: Duty Cycle

**Duty Cycle**(DC)이란 한 사이클 내에서 High Voltage가 차지하는 비율을 의미한다. [StackExchange](<https://raspberrypi.stackexchange.com/questions/108111/what-is-the-relationship-between-angle-and-servo-motor-duty-cycle-how-do-i-impl>)에 따르면 대부분의 서보모터는 $$1000\mu s \backsim 2000\mu s$$의 Pulse에 반응하며, $$1500\mu s$$를 기준으로 $$+10\mu s$$당 시계방향으로 1도씩, $$-10\mu s$$당 반시계 방향으로 1도씩 가리키는 방향이 달라진다고 한다. 개별 서보 모터의 사양에 따라 다르겠지만 각각의 경우에 대략적인 각도를 계산해보면 다음과 같다. 참고로 50Hz는 1초당 50사이클을 의미하므로 1사이클의 길이는 20ms이 된다.

|Frequency|Duty Cycle(%)|Pulse|Angle|
|:----:|:---:|:---:|:---:|
| 50Hz | 0 | $$0\mu s$$ | Out of Range |
| 50Hz | 2.5 | $$500\mu s$$ | Out of Range |
| 50Hz | 5 | $$1000\mu s$$ |0 |
| 50Hz | 7.5 | $$1500\mu s$$ | +50 |
| 50Hz | 10 | $$2000\mu s$$ | +100 |
| 50Hz | 50 | $$10000\mu s$$ | Out of Range |

Futaba s3003를 가지고 실험을 해보니 각 DC마다 대략적으로 다음과 같이 움직이는 것을 확인할 수 있었다.

|Frequency|Duty Cycle(%)|Pulse|Angle|
|:----:|:---:|:---:|:---:|
| 50Hz | 0 | $$0\mu s$$ | Out of Range |
| 50Hz | 2.5 | $$500\mu s$$ | -90|
| 50Hz | 5 | $$1000\mu s$$ | -45 |
| 50Hz | 7.5 | $$1500\mu s$$ | 0 |
| 50Hz | 10 | $$2000\mu s$$ | +45 |
| 50Hz | 12.5 | $$2500\mu s$$ | +90 |
| 50Hz | 13 | $$2600\mu s$$ | Out of Range |
| 50Hz | 50 | $$10000\mu s$$ | Out of Range |

예를 들어 아래와 같이 코드를 작성하면

```python
p.start(5)

try:
  while True:
    p.ChangeDutyCycle(5)
    time.sleep(0.5)
    p.ChangeDutyCycle(10)
    time.sleep(0.5)
except KeyboardInterrupt:
  p.stop()
```

-45도와 +45도 사이를 왔다갔다하게 되고, 아래와 같이 작성하면 

```python
p.start(0)

p.ChangeDutyCycle(5)
time.sleep(0.5)

try:
  while True:
    p.ChangeDutyCycle(7.5)
    time.sleep(0.5)
except KeyboardInterrupt:
  p.stop()
```

-45도에서 0도로 한 번 움직인 뒤(시계방향 45도) 더 이상 움직이지 않게 된다(DC=7.5로 고정). 즉 Pulse 값은 가리키는 방향을 의미하지 움직이는 각도의 크기를 의미하지 않는다. 참고로 PWM을 시작할 때 다음과 같이 Parameter를 전달하게 되는데 초기 DC 값을 0으로 설정하겠다는 것을 뜻한다.

```python
p.start(0)
```

그리고 서보모터의 동작은 다음 코드로 멈출 수 있다.

```python
p.stop()
```

#### Clean Up

마지막으로 다음 코드를 통해 GPIO를 더 이상 사용하지 않겠다고 해주어야 한다.

```python
GPIO.cleanup()
```
