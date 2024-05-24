"""
Code for parsing the activitygen xml and extracting out all of the start and end points
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from collections import defaultdict

class Person:

    def __init__(self, id, departure_time):

        self.departure_time = departure_time
        self.id = id
        self.stop_points = []
    
    def add_stop(self, stop_point: StopPoint):

        self.stop_points.append(stop_point)


class StopPoint:

    def __init__(self, edge, position, duration):
        self.edge = edge
        self.position = position
        self.duration = duration


def parse_edge_stop_points(xml_path):
    context = ET.iterparse(xml_path, events=("start", "end"))
    context = iter(context)
    ev, root = next(context)

    # pbar = tqdm(total=total_timesteps)
    # progress = 0

    agents = {}

    for ev, el in context:
        if ev == 'start' and el.tag == 'person':
            agent_id = el.attrib['id']
            agent = Person(agent_id, el.attrib['depart'])
            agents[agent_id] = agent
            arrival_pos = None
            for sub in el:
                if 'arrivalPos' in sub.attrib:
                    arrival_pos = sub.attrib['arrivalPos']
                if sub.tag == 'stop':
                    attrib = sub.attrib
                    stop_point = StopPoint(
                        attrib['lane'], arrival_pos, attrib['duration'])
                    agent.add_stop(stop_point)
    return agents


if __name__ == "__main__":

    xml_path = '/home/local/KHQ/cole.hill/Documents/HAYSTAC/haystac_eval/osm_activitygen.merged.rou.xml'
    agents = parse_edge_stop_points(xml_path)
    print(agents)
