from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import streamlit as st

# دالة للاتصال بـ Google Drive
def authenticate_drive():
    # جلب البيانات السرية من إعدادات Streamlit
    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)
    return service

# دالة لرفع الملف
def upload_to_drive(file_buffer, file_name, folder_id):
    service = authenticate_drive()
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id] # ضع الـ ID الخاص بمجلد Drive هنا
    }
    
    media = MediaIoBaseUpload(io.BytesIO(file_buffer.getvalue()), 
                              mimetype=file_buffer.type,
                              resumable=True)
                              
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    # جعل الملف متاحاً للجميع للقراءة
    service.permissions().create(
        fileId=uploaded_file.get('id'),
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    return uploaded_file.get('webViewLink') # يُرجع رابط قراءة الكتاب
