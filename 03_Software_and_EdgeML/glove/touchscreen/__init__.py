"""
The Winstar touchscreen: the home screen ([Free] [Lesson] [Recalibrate]) and free mode.

  app.py      the home screen and what each button does; main.py starts it
  screens.py  every screen, drawn as a 128x64 picture with its touch buttons
  lcd.py      puts the pictures on the Winstar, sending only the bytes that changed
  panel.py    reads the 4-wire resistive touch panel and turns touches into taps
  sim.py      main.py --mock: the touchscreen in a window (or the terminal) on any computer
"""
