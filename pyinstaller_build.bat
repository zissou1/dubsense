
pushd %~dp0
pyinstaller --onefile --noconsole --additional-hooks-dir=. --icon=dubsense.ico --add-binary "C:\Users\arnqv\AppData\Local\Programs\Python\Python310\Lib\site-packages\paddle\libs\mklml.dll;." dubsense.py
popd
@pause