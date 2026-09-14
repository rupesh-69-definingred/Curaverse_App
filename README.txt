CURAVERSE - Corrected Same-UI Prototype
=======================================

Start:
  python app.py

Open:
  http://127.0.0.1:5000

Demo OTP:
  123456

Patients:
  No demo/fixed patient is pre-loaded. The app starts with an empty
  patient list — add patients from "Add Patient" and every added
  patient gets their own Image Processing workflow.

Flow:
  OTP Login -> Dashboard -> Patients -> Add Patient -> Medical History ->
  Symptoms -> Clinical Examination -> Image Processing -> Reports ->
  Appointments -> Analytics -> Settings

The CSS is intentionally kept inside a Python triple-quoted string in app.py.
Do not paste CSS outside the CSS = """ ... """ block.

Image processing is a prototype visual segmentation workflow using grayscale
preprocessing and K-Means clustering with K=3. It must not be presented as an
automated clinical diagnosis.
