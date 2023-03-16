# yolov7-feature-map-visualization
my own yolov7 feature-map visualization code.

#   ReadMe <viz-filter-feature-map.py>   #
Example Command

python viz-filter-feature-map.py --model {YOUR MODEL'S WEIGHT(.pt) FILE} --cfg {CFG FILE(.yaml) OF YOUR MODEL}--image {YOUR IMG} --name {OUTPUT FOLDER NAME, CREATED AT visualize-filter/feature-map}

python viz-filter-feature-map.py --model ./Please_Remember_Me/train/yolov7_Argo_origin4/weights/epoch_029.pt --cfg ./cfg/training/yolov7-tiny.yaml --image ./visualize-filter/src/cute-cat.jpg --name tiny-original


<!Caution!>
1. .py should be placed in yolov7 folder

2. Last Detect Layer is not visualized. it continuously occurs error, so i skipped last part.
