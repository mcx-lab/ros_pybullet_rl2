#!/usr/bin/env python
import rospy

import dynamic_reconfigure.client

def callback(config):
    rospy.loginfo("""Config set to {inflation_human_cost}, {obstacle_human_cost}""".format(**config))

if __name__ == "__main__":
    rospy.init_node("dynamic_cost_client")

    client_global_obstacle = dynamic_reconfigure.client.Client("/move_base/global_costmap/obstacle_layer", timeout=30, config_callback=callback)
    client_local_obstacle = dynamic_reconfigure.client.Client("/move_base/local_costmap/obstacle_layer", timeout=30, config_callback=callback)
    client_global_inflation = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer", timeout=30, config_callback=callback)
    client_local_inflation = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer", timeout=30, config_callback=callback)

    # r = rospy.Rate(1)
    human_cost = 100
    params = {"inflation_human_cost": human_cost, "obstacle_human_cost": human_cost}
    
    client_global_obstacle.update_configuration(params)
    client_local_obstacle.update_configuration(params)
    client_global_inflation.update_configuration(params)
    client_local_inflation.update_configuration(params)