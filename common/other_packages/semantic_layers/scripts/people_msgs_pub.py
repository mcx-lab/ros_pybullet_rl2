#!/usr/bin/env python
import rospy
from semantic_layers.msg import People, Person
import random

if __name__ == "__main__":
    try:
        pub = rospy.Publisher('people', People)
        rospy.init_node('people_pub', anonymous=True)
        r = rospy.Rate(0.5)

        while not rospy.is_shutdown():
            msg = People()
            person1 = Person()
            person1.name = "person1"
            
            person2 = Person()
            person2.name = "person2"
            
            person3 = Person()
            person3.name = "person3"

            # person1.position.x = random.uniform(-2.0, 2.0)
            # person1.position.y = random.uniform(-2.0, 2.0)
            # person1.velocity.x = random.uniform(-1.0, 1.0)
            # person1.velocity.y = random.uniform(-1.0, 1.0)

            # person2.position.x = random.uniform(-2.0, 2.0)
            # person2.position.y = random.uniform(-2.0, 2.0)
            # person2.velocity.x = random.uniform(-1.0, 1.0)
            # person2.velocity.y = random.uniform(-1.0, 1.0)
            person1.position.x = -1.0
            person1.position.y = 0.0
            person1.velocity.x = 0.0
            person1.velocity.y = 0.0
            person1.reliability = 1.0
            person1.cost = 100

            person2.position.x = 0.0
            person2.position.y = -1.5
            person2.velocity.x = 0.3
            person2.velocity.y = 0.2
            person2.reliability = 1.0
            person2.cost = 200

            person3.position.x = 1.0
            person3.position.y = 1.0
            person3.velocity.x = 0.1
            person3.velocity.y = -0.6
            person3.reliability = 1.0
            person3.cost = 200
            
            msg.people = [person1, person2, person3]
            msg.header.frame_id = 'map'

            rospy.loginfo(msg)
            pub.publish(msg)
            r.sleep()

    except rospy.ROSInterruptException: pass
