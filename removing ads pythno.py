from lxml.etree import tostring
import lxml.html
import requests
import mechanicalsoup
from adremover import AdRemover
from bs4 import BeautifulSoup
import re
from xml.etree import ElementTree
url = 'https://www.good.is/articles/beekeeping-community-learning'  # replace it with a url you want to apply the rules to
rule_urls = ['https://easylist-downloads.adblockplus.org/ruadlist+easylist.txt',
             'https://filters.adtidy.org/extension/chromium/filters/1.txt']

rule_files = [url.rpartition('/')[-1] for url in rule_urls]


# download files containing rules
for rule_url, rule_file in zip(rule_urls, rule_files):
    r = requests.get(rule_url)
    with open(rule_file, 'w', encoding="utf-8") as f:
        print(r.text, file=f)


remover = AdRemover(*rule_files)
print(remover)
html = requests.get(url).text
document = lxml.html.document_fromstring(html)
remover.remove_ads(document)
clean_html = tostring(document).decode("utf-8")
print(clean_html)