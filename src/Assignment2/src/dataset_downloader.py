import os
import time
import requests
import random
from pytube import YouTube
from PIL import Image
from io import BytesIO
from google_images_search import GoogleImagesSearch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import cv2

# Create directories for dataset
os.makedirs('dataset/robot/train', exist_ok=True)
os.makedirs('dataset/robot/test', exist_ok=True)
os.makedirs('dataset/person/train', exist_ok=True)
os.makedirs('dataset/person/test', exist_ok=True)

# Function to download frames from YouTube videos
def download_frames_from_youtube(video_urls, output_dir, num_frames=180):
    frames_collected = 0
    frame_indices = []
    
    for url in video_urls:
        if frames_collected >= num_frames:
            break
            
        try:
            # Download video
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            temp_file = stream.download(output_path='temp', filename='temp_video')
            
            # Open video and get information
            cap = cv2.VideoCapture(temp_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Generate random frames to extract (avoiding consecutive frames)
            available_frames = min(total_frames, 100)  # Limit per video
            random_frames = sorted(random.sample(range(0, total_frames), min(available_frames, num_frames - frames_collected)))
            
            # Extract frames
            for i, frame_idx in enumerate(random_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Save frame
                    frames_collected += 1
                    save_path = os.path.join(output_dir, f"robot{frames_collected}.png")
                    cv2.imwrite(save_path, frame)
                    frame_indices.append(frame_idx)
                    
                    # Break if collected enough frames
                    if frames_collected >= num_frames:
                        break
            
            # Release video
            cap.release()
            os.remove(temp_file)
            
            # Don't hammer YouTube servers
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing video {url}: {e}")
    
    return frame_indices

# Function to download images from Google
def download_person_images(num_images=180):
    # Set up Chrome driver
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    
    search_terms = [
        "person full body", "people standing", "human full body", 
        "person portrait", "diverse people", "humans crowd",
        "person walking", "people different ages", "diverse humans"
    ]
    
    images_collected = 0
    
    for search_term in search_terms:
        if images_collected >= num_images:
            break
            
        # Search Google Images
        driver.get(f"https://www.google.com/search?q={search_term.replace(' ', '+')}&tbm=isch")
        time.sleep(2)  # Let the page load
        
        # Scroll down to load more images
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Find image elements
        img_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
        img_urls = []
        
        # Extract image URLs
        for img in img_elements:
            try:
                img.click()
                time.sleep(1)
                
                # Try different ways to get the large image URL
                large_img = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
                if large_img:
                    src = large_img[0].get_attribute("src")
                    if src and src.startswith("http") and src not in img_urls:
                        img_urls.append(src)
            except:
                pass
        
        # Download each image
        for url in img_urls:
            if images_collected >= num_images:
                break
                
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content))
                
                # Save the image
                images_collected += 1
                save_path = f"dataset/person/person{images_collected}.png"
                img.save(save_path)
                
                # Avoid hammering the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error downloading image {url}: {e}")
    
    driver.quit()
    return images_collected

# Robot video URLs from the provided list
robot_videos = [
    "https://www.youtube.com/watch?v=gtJXGo8WEMY",
    "https://www.youtube.com/watch?v=hKLC0Vz1GmM",
    "https://www.youtube.com/watch?v=6ldHWWHfeBc",
    "https://www.youtube.com/watch?v=WJKc56uUuF8",
    "https://www.youtube.com/watch?v=RG205OwGdSg",
    "https://www.youtube.com/watch?v=yVdB_0ry53o",
    "https://www.youtube.com/watch?v=TWNvSHpMrSM",
    "https://www.youtube.com/watch?v=UsmBD2_3FH8",
    "https://www.youtube.com/watch?v=WGKo_6IkFBY",
    "https://www.youtube.com/watch?v=G6xE7uWt6Fo",
    "https://www.youtube.com/watch?v=DrNcXgoFv20",
    "https://www.youtube.com/watch?v=cpraXaw7dyc",
    "https://www.youtube.com/watch?v=raYWbqbZbmc",
    "https://www.youtube.com/watch?v=F_7IPm7f1vI"
]

# Create temporary directory
os.makedirs('temp', exist_ok=True)

print("Downloading robot images from YouTube videos...")
robot_frames = download_frames_from_youtube(robot_videos, 'dataset/robot', num_frames=180)
print(f"Downloaded {len(robot_frames)} robot images")

print("Downloading person images from Google...")
person_images = download_person_images(num_frames=180)
print(f"Downloaded {person_images} person images")

# Split data into train and test sets
def split_dataset(class_name, train_count=150):
    source_dir = f"dataset/{class_name}"
    train_dir = f"dataset/{class_name}/train"
    test_dir = f"dataset/{class_name}/test"
    
    # Get all images
    all_images = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    random.shuffle(all_images)
    
    # Split images
    train_images = all_images[:train_count]
    test_images = all_images[train_count:]
    
    # Move images to appropriate directories
    for i, img in enumerate(train_images):
        os.rename(
            os.path.join(source_dir, img),
            os.path.join(train_dir, f"{class_name}{i+1}.png")
        )
    
    for i, img in enumerate(test_images):
        os.rename(
            os.path.join(source_dir, img),
            os.path.join(test_dir, f"{class_name}{i+1}.png")
        )

# Split datasets
print("Splitting datasets into training and testing sets...")
split_dataset('robot')
split_dataset('person')

# Remove temporary directory
os.system('rm -rf temp')

print("Dataset preparation complete!")
print("Training sets: 150 robot images, 150 person images")
print("Testing sets: 30 robot images, 30 person images")