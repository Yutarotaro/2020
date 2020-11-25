cd ~/research/ml/object_detection_tools

python3 scripts/object_detection_tf2_for_image.py -l='./models/coco-labels-paper.txt' -m='./models/efficientdet_d0_coco17_tpu-32/saved_model/' -me='dia_V' -from='120' -to="170" 
