import requests
import os

# Client per SsdWebApi
# Lanciare con `python3 client.py`

def get_all_quotazioni():
  r = requests.get('https://localhost:5001/api/Quotazione', verify=False)
  quotazioni = r.json()
  for quotazione in quotazioni:
    print(quotazione)
    print('----------')

# change working directory to script path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# get quotazioni
print("--- GET ALL ---")
get_all_quotazioni()

# insert quotazione
#print("--- POST ---")
#requests.post('http://localhost:5000/api/Quotazione', json={"id":10,"anno":2020,"serie":"C"}, verify=False)
#get_all_quotazioni()

# remove reminder
#print("--- DELETE ---")
#id=10
#requests.delete('http://localhost:5000/api/Quotazione/'+str(id), verify=False)
#get_all_quotazioni()
