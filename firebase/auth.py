import pyrebase
import streamlit as st
# Firebase configuration
firebaseConfig = {
    "apiKey": "AIzaSyClPElKtYcoBYbapv-fxwsPMT-JZoxb71g",
    "authDomain": "xai-risk-framework.firebaseapp.com",
    "databaseURL": "https://xai-risk-framework-default-rtdb.firebaseio.com",
    "projectId": "xai-risk-framework",
    "storageBucket": "xai-risk-framework.firebasestorage.app",
    "messagingSenderId": "452409742218",
    "appId": "1:452409742218:web:c936b910b59310d838ae73"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()


def register_user(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return {"success": True, "user": user}
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            return {"success": False, "error": "Email already registered"}
        elif "WEAK_PASSWORD" in error_message:
            return {"success": False, "error": "Password should be at least 6 characters"}
        elif "INVALID_EMAIL" in error_message:
            return {"success": False, "error": "Invalid email address"}
        else:
            return {"success": False, "error": "Registration failed. Try again."}


def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return {"success": True, "user": user}
    except Exception as e:
        error_message = str(e)
        if "INVALID_LOGIN_CREDENTIALS" in error_message:
            return {"success": False, "error": "Invalid email or password"}
        elif "INVALID_EMAIL" in error_message:
            return {"success": False, "error": "Invalid email address"}
        else:
            return {"success": False, "error": "Login failed. Try again."}


def reset_password(email):
    try:
        auth.send_password_reset_email(email)
        return {"success": True}
    except:
        return {"success": False, "error": "Could not send reset email"}
