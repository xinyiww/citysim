import rosbag
import numpy as np
import matplotlib.pyplot as plt

# bag = rosbag.Bag('sample_center_lane.bag')
bag = rosbag.Bag('bags/sample.bag')



for topic, msg, t in bag.read_messages():
    print(f"Topic: {topic}, Message: {msg}, Time: {t}")

bag.close()