from icrawler.builtin import BingImageCrawler
import os

# Step 2: Set up download directory
save_dir = 'humanoid_robot_images'
os.makedirs(save_dir, exist_ok=True)

# Step 3: Use BingImageCrawler to fetch images
crawler = BingImageCrawler(storage={'root_dir': save_dir})
crawler.crawl(keyword='humanoid robot', max_num=200)

print(f"Downloaded 100 images to: {save_dir}")
