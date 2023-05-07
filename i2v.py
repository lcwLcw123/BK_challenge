import os
import cv2
from tqdm import tqdm

def create_video_from_images(input_folder, output_video, fps=30):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    i=1
    for img_file in tqdm(image_files):
        img_path = os.path.join(input_folder, str(i)+'.jpg')
        img = cv2.imread(img_path)
        video_writer.write(img)
        i= i+1

    video_writer.release()

def main():

    input_folder = "./result/"
    output_video = "output_video.mp4"
    fps = 26
    create_video_from_images(input_folder, output_video, fps)

main()
