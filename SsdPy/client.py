import requests
import os

# Client per SsdWebApi
# Lanciare con `python3 client.py`

def get_all_stagioni():
  r = requests.get('https://localhost:5001/api/Stagione', verify=False)
  stagioni = r.json()
  for stagione in stagioni:
    print(stagione)
    print('----------')

# change working directory to script path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# get stagioni
print("--- GET ALL ---")
get_all_stagioni()

# insert stagione
print("--- POST ---")
requests.post('http://localhost:5000/api/Stagione', json={"id":10,"anno":2020,"serie":"C"}, verify=False)
get_all_stagioni()

# remove reminder
print("--- DELETE ---")
id=10
requests.delete('http://localhost:5000/api/Stagione/'+str(id), verify=False)
get_all_stagioni()
