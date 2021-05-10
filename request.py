import requests
import string
import json
import re

BASE = "http://81132d2ff217.ngrok.io"

input_urls = input("URLs (seperated by comma): ")
listing = input("Listing name: ")
url_list = []
for i in re.split(",\s*", input_urls):
    url_list.append(i)

# print('URL list: ' + str(url_list))
# try:
# res = requests.post(BASE + '/duplicate_check/', json={"listing": listing, "urls":url_list})
res = requests.post(BASE + '/add/', json={"listing": listing, "urls":url_list})
print("Successful")
parsed_json_res = res.json()
print(json.dumps(parsed_json_res, indent=4, sort_keys=True))
# except:
    # print("Failed to send request")
    
