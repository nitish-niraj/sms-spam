@echo off
cd /d E:\SMS
REM Note: Set HF_TOKEN environment variable if needed for authenticated model downloads
REM set HF_TOKEN=your_hugging_face_token_here
echo ============================================================
echo Starting SMS Spam Detection Training
echo ============================================================
echo.
echo This will take 2-3 hours on CPU. Please do not close this window.
echo Training progress will be displayed below...
echo.
echo ============================================================
echo.
python sms_spam_bert.py
echo.
echo ============================================================
echo Training Complete!
echo ============================================================
pause
