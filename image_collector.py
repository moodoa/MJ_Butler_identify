import os
import time
import requests

from cv2 import cv2 as cv 
from PIL import Image
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class img_collector():
    def __init__(self):
        pass
    
    def get_img(self, player):
        os.mkdir(f"{player}/")
        chrome_options = Options() 
        chrome_options.add_argument('--headless') 
        driver = webdriver.Chrome(chrome_options=chrome_options)
        driver.get(f"https://www.google.com/search?q={player}&hl=zh-TW&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiDgrGg7MnrAhVPxIsBHbQYBTgQ_AUoAXoECBwQAw&biw=1536&bih=722")
        for pullrange in range(1,5):
            js = "var q=document.documentElement.scrollTop={}".format(pullrange*10000)
            driver.execute_script(js)
            time.sleep(1)
        page_source = driver.page_source
        driver.close()
        soup = BeautifulSoup(page_source, "html.parser")
        file_name = 0
        for element in soup.select("img"):
            try:
                with open(f"{player}/{str(file_name)}.jpg", "wb") as file:
                    file.write(requests.get(element.get("src")).content)
                    file_name += 1
            except:
                pass
        return "ok"

    def extract_face(self, player):
        face_cascade = cv.CascadeClassifier(r"C:\Users\User\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
        destination = f"{player}_faces/"
        os.mkdir(destination)
        for file in os.listdir(f"{player}/"):
            img = Image.open(f"{player}/{file}")
            image_matrix = cv.imread(f"{player}/{file}")
            if len(face_cascade.detectMultiScale(image_matrix, 1.3, 5)) == 1:
                x, y, width, height = face_cascade.detectMultiScale(image_matrix, 1.3, 5)[0]
                img_crop = img.crop((x, y, x+width, y+height)).resize((64, 64))
                img_crop.save(destination+file)
        return "ok"