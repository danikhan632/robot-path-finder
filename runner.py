import os
import json
import time
import numpy as np
import anki_vector
from PIL import Image
from io import BytesIO

#from transformers import ViTFeatureExtractor, ViTForImageClassification,BeitFeatureExtractor, BeitForImageClassification,SegformerFeatureExtractor, SegformerForImageClassification
# from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from anki_vector.util import Pose, degrees, distance_mm, speed_mmps,distance_inches
from anki_vector import behavior
import torch
def move_robot(robot, dist):
    robot.behavior.drive_straight(distance_inches(dist* 12), speed_mmps(100))
    # show_viewer=True, show_3d_viewer=True
    
def get_camera_obj(feature_extractor,model, image,device,tokenizer):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds[0]


def main():
    poses=[]
  
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with open('data.json', 'r') as infile:
        poses = json.load(infile)
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(show_viewer=True) as robot:

        # curr_pose = starting_pose
        robot.camera.init_camera_feed()
        robot.conn.request_control()
        robot.motors.set_head_motor(0)
        time.sleep(2)

        for i in range(0,len(poses)-1):
            image = robot.camera.latest_image.raw_image
            robot.behavior.say_text(get_camera_obj(feature_extractor,model, image,device,tokenizer))
            curr_point= np.array([poses[i][0],poses[i][1]])
            next_point= np.array([poses[i+1][0],poses[i+1][1]])
            dist=np.linalg.norm(next_point-curr_point)
            print("moving ", round(abs(dist),3) ," inches in direction",poses[i][2] )
            robot.behavior.turn_in_place(degrees(poses[i][2]))
            move_robot(robot, abs(dist))
        exit()



if __name__ == '__main__':
    try:
        main()
    except anki_vector.exceptions.VectorTimeoutException:
        try:
            main()
        except anki_vector.exceptions.VectorTimeoutException:
            try:
                main()
            except anki_vector.exceptions.VectorTimeoutException:
                try:
                    main()
                except anki_vector.exceptions.VectorTimeoutException:
                    try:
                        main()
                    except anki_vector.exceptions.VectorTimeoutException:
                        try:
                            main()
                        except anki_vector.exceptions.VectorTimeoutException:
                            try:
                                main()
                            except anki_vector.exceptions.VectorTimeoutException:
                                try:
                                    main()
                                except anki_vector.exceptions.VectorTimeoutException:
                                    main()
        
        
        
        # buffer = BytesIO()
        # print(type(image.raw_image))
        # # pil_image = Image.frombytes(mode='RGB',size=(640,360),  data=image.raw_image)
        # image.raw_image.save('image.png', format='PNG')
        # # image.raw_image.show()
        
            # feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    # model = ViTForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    # feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    # model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b1")
    # model = SegformerForImageClassification.from_pretrained("nvidia/mit-b1")
    
    #     inputs = feature_extractor(images=image.raw_image, return_tensors="pt")
    # outputs = model(**inputs)
    # logits = outputs.logits
    # # model predicts one of the 1000 ImageNet classes
    # predicted_class_idx = logits.argmax(-1).item()
      # feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-384-22k-1k")
    # model = ConvNextForImageClassification.from_pretrained("facebook/convnext-xlarge-384-22k-1k")
    
        # inputs = feature_extractor(image, return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    # predicted_label = logits.argmax(-1).item()
    # pred=model.config.id2label[predicted_label]
    # print(pred)