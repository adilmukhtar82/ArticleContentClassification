from bs4 import BeautifulSoup
import requests

import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile
import re

Write_Excel_Filename = 'res-classify_links.xlsx'
Read_Excel_Filename = 'classify_links.xlsx'
Sheet_Name = 'Sheet1'
Sponsored_Tag_List = ["Story from" , "Sponsored by", "BrandPost", "Provided by:"]
#Sponsored_Tag_List = ["STORY FROM" , "SPONsored by", "BranDPost", "Provided by:"]
Links_List = []
Status_List = []

df = pd.read_excel(Read_Excel_Filename, sheetname=Sheet_Name)

for i in df.index:
    page_link = df['Links'][i]
    Links_List.append(page_link)
    
    #page_link ='https://www.techrepublic.com/resource-library/casestudies/case-study-iot-monetization-at-current-powered-by-ge/'
    # fetch the content from url
    page_response = requests.get(page_link, timeout=5)
    # parse html
    page_content = BeautifulSoup(page_response.content, "html.parser")
    page_content = page_content.prettify()

    #status = any(tag in page_content for tag in Sponsored_Tag_List)
    status = any(re.search(tag, page_content, re.IGNORECASE) for tag in Sponsored_Tag_List)
    print(str(i) + ' : ' + str(status))

    if status:
        Status_List.append('Sponsored')
    else:
        Status_List.append('Non-Sponsored')
        

StatusFrame = pd.DataFrame({'Links':Links_List, 'Status':Status_List})
writer = ExcelWriter(Write_Excel_Filename,engine='openpyxl')
StatusFrame.to_excel(writer,Sheet_Name,index=False)
writer.save()

'''
if "Story from" not in page_content:
    print ("False")
else:
    print ("True")
'''
