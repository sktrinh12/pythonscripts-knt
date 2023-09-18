import requests

fp = '/Users/spencer.trinhkinnate.com/Downloads/Data_to_upload_to_DM/'
url = 'http://127.0.0.1:8000/upload'
files = [('files', open(f'{fp}PLM-FT008460-03.pdf', 'rb')), ('files', open(f'{fp}TGA-FT008460-03.pdf', 'rb'))]
resp = requests.post(url=url, files=files)
print(resp.json())
