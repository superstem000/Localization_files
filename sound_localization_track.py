import json
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_position(doa1, doa2, d):
    doa1_rad = np.radians(doa1 + 90)
    doa2_rad = np.radians(doa2 + 90)
    
    if doa1 == 0 or doa1 == 180:
        m1 = np.inf
    else:
        m1 = np.tan(doa1_rad)

    if doa2 == 0 or doa2 == 180:
        m2 = np.inf
    else:
        m2 = np.tan(doa2_rad)
    
    if m1 == np.inf and m2 == np.inf:
        return None

    if m1 == np.inf:
        intersection_x = d / 2
        intersection_y = m2 * (intersection_x + d / 2)
    elif m2 == np.inf:
        intersection_x = -d / 2
        intersection_y = m1 * (intersection_x - d / 2)
    elif m1 == m2:
        intersection_x = 0
        intersection_y = 0
    else:
        intersection_x = d/2 * (m2 + m1) / (-m2 + m1)
        intersection_y = m2 * (intersection_x + d / 2)

    return intersection_x, intersection_y

def find_closest_doa(data, target_time):
    closest_entry = min(data, key=lambda x: abs(x['record_time'] - target_time))
    return closest_entry['doa']

def track_localization(data_1, data_2, d):
    positions = []
    times = []
    
    combined_data = sorted(data_1 + data_2, key=lambda x: x['record_time'])

    for entry in combined_data:
        current_time = entry['record_time']
        current_doa1 = find_closest_doa(data_1, current_time)
        current_doa2 = find_closest_doa(data_2, current_time)
        
        position = calculate_position(current_doa1, current_doa2, d)
        if position is not None:
            positions.append(position)
            times.append(current_time)
    
    return positions, times

class LiveTracker:
    def __init__(self, positions, d, zone_centers=None):
        self.positions = positions
        self.d = d
        
        if zone_centers is None:
            self.zone_centers = [np.array([-100, 100]), np.array([100, 100])]
        else:
            self.zone_centers = zone_centers
            
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.trail_length = 20  # Number of previous positions to show
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3, label='Recent Path')
        self.current_pos, = self.ax.plot([], [], 'bo', markersize=10, label='Current Position')
        
        self.ax.scatter([-d/2, d/2], [0, 0], color='red', label='Microphones')
        self.ax.text(-d/2, 0.1, 'Mic 1', color='red', ha='center')
        self.ax.text(d/2, 0.1, 'Mic 2', color='red', ha='center')
        
        # Calculate and plot trapezoids
        array_center = np.array([0, 0])
        self.trapezoids = []
        for i, zone_center in enumerate(self.zone_centers):
            trapezoid = self.calculate_trapezoid(zone_center, array_center)
            trap_plot, = self.ax.plot(
                [p[0] for p in np.vstack((trapezoid, trapezoid[0]))],
                [p[1] for p in np.vstack((trapezoid, trapezoid[0]))],
                'g-', alpha=0.3, label=f'Detection Zone {i+1}'
            )
            self.trapezoids.append((trapezoid, trap_plot))
        
        self.ax.grid(True)
        self.ax.set_xlabel('X Position (cm)')
        self.ax.set_ylabel('Y Position (cm)')
        self.ax.set_title('Live Sound Source Tracking')
        
        x_min, x_max = -600, 600
        y_min, y_max = -600, 600
        padding = max(abs(x_max - x_min), abs(y_max - y_min)) * 0.1
        
        self.ax.set_xlim(x_min - padding, x_max + padding)
        self.ax.set_ylim(y_min - padding, y_max + padding)
        
        self.ax.legend()

    def calculate_trapezoid(self, zone_center, array_center, front_width=90, back_width=150, height=150):
        # Calculate angle from array center to zone center
        direction = zone_center - array_center
        angle = np.arctan2(direction[1], direction[0]) - np.pi / 2
        
        # Define trapezoid with narrow side facing array center
        trapezoid_local = np.array([
            [-front_width/2, 0],
            [front_width/2, 0],
            [back_width/2, height],
            [-back_width/2, height]
        ])
        
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        trapezoid_centered = trapezoid_local - np.array([0, height/2])
        
        trapezoid_global = np.dot(trapezoid_centered, rotation_matrix.T) + zone_center
        
        return trapezoid_global

    def is_point_in_trapezoid(self, point, trapezoid):

        n = len(trapezoid)
        inside = False
        
        for i in range(n):
            j = (i + 1) % n
            if ((trapezoid[i][1] > point[1]) != (trapezoid[j][1] > point[1]) and
                point[0] < (trapezoid[j][0] - trapezoid[i][0]) * 
                (point[1] - trapezoid[i][1]) / 
                (trapezoid[j][1] - trapezoid[i][1]) + trapezoid[i][0]):
                inside = not inside
                
        return inside

    def update(self, frame):
        start_idx = max(0, frame - self.trail_length)
        x_trail = [pos[0] for pos in self.positions[start_idx:frame+1]]
        y_trail = [pos[1] for pos in self.positions[start_idx:frame+1]]
        self.trail.set_data(x_trail, y_trail)
        
        # Update current position
        if frame < len(self.positions):
            current_pos = self.positions[frame]
            self.current_pos.set_data([current_pos[0]], [current_pos[1]])
            
            containing_trapezoids = []
            
            for i, (trapezoid, trap_plot) in enumerate(self.trapezoids):
                if self.is_point_in_trapezoid(current_pos, trapezoid):
                    # Calculate distance to zone center
                    distance = np.linalg.norm(np.array(current_pos) - self.zone_centers[i])
                    containing_trapezoids.append((i, distance, trap_plot))
            
            # Reset all trapezoids to default color
            for _, trap_plot in self.trapezoids:
                trap_plot.set_color('g')
                trap_plot.set_alpha(0.3)
            
            # If point is in any trapezoids, color only the closest one
            if containing_trapezoids:
                containing_trapezoids.sort(key=lambda x: x[1])
                closest_trapezoid = containing_trapezoids[0]
                closest_trapezoid[2].set_color('r')
                closest_trapezoid[2].set_alpha(0.4)
        
        return (self.trail, self.current_pos, 
                *[trap_plot for _, trap_plot in self.trapezoids])

def animate_trajectory(positions, d):

    with open("mic2ID.json", 'r') as f:
        mic2_data = json.load(f)
    with open("mic3ID.json", 'r') as f:
        mic3_data = json.load(f)

    zone_centers = []
    
    for person_key in mic2_data:
        person_id = mic2_data[person_key]["ID"]
        
        matching_person = next(
            (p for p in mic3_data.values() if p["ID"] == person_id),
            None
        )
        
        if matching_person:
            doa1 = mic2_data[person_key]["doa"]
            doa2 = matching_person["doa"]
            position = calculate_position(doa1, doa2, d)
            zone_centers.append(position)


    tracker = LiveTracker(positions, d, zone_centers = zone_centers)
    
    anim = FuncAnimation(tracker.fig, tracker.update, 
                        frames=len(positions),
                        interval=20, #50 default
                        blit=True,
                        repeat=False)
    
    plt.show()
    
    return anim

# Main function to process the data and display the animation
def process_and_animate(data_1, data_2, d):
    positions, times = track_localization(data_1, data_2, d)
    return animate_trajectory(positions, d)


def group_files_by_identifier(folder_path):
    grouped_files = {"DOA_2": [], "DOA_3": []}
    
    for filename in os.listdir(folder_path):
        if re.search(r"DOA_2", filename):
            grouped_files["DOA_2"].append(os.path.join(folder_path, filename))
        elif re.search(r"DOA_3", filename):
            grouped_files["DOA_3"].append(os.path.join(folder_path, filename))
    
    return grouped_files

def load_and_combine_files(file_list):
    combined_data = []
    for file in file_list:
        with open(file) as f:
            combined_data.extend(json.load(f))
    return combined_data

def process_all_files(folder_path, d):
    grouped_files = group_files_by_identifier(folder_path)
    
    data_1 = load_and_combine_files(grouped_files["DOA_2"])
    data_2 = load_and_combine_files(grouped_files["DOA_3"])
    
    return process_and_animate(data_1, data_2, d)



# Run the animation
distance_between_mics = 50  # Adjust as needed
anim = process_all_files(".", distance_between_mics)