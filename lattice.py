import networkx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from random import uniform, randint
import functools
import imageio
import sys
import math
import matplotlib.animation as animation

class Lattice:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.points = self.create_lattice(m, n)

    def create_lattice(self, m, n):
        self.shortest_time.cache_clear()
        self.time_and_path.cache_clear()
        points = {}
        #Create points
        for x in range(m + 1):
            for y in range(n + 1):
                points[(x, y)] = uniform(0, 1)
        points[(0, 0)] = 0
        return points

    def update_lattice(self):
        self.shortest_time.cache_clear()
        self.time_and_path.cache_clear()
        m = randint(0, self.m)
        n = randint(0, self.n)
        point = (m, n)
        self.points[point] = uniform(0, 1)
        return point

    def max_distance(self, path1, path2):
        max = 0
        for line in range(self.n + 1):
            dist = distance(path1[line], path2[line])
            if dist > max:
                max = dist
        return max

    def time_to_flip(self):
        orig_path = self.time_and_path((0, 0), (self.m, self.n))[1]
        max_dist = 0
        num_changes = 0
        while max_dist < self.n ** (2 / 3):
            self.update_lattice()
            new_path = self.time_and_path((0, 0), (self.m, self.n))[1]
            max_dist = self.max_distance(orig_path, new_path)
            num_changes += 1
        return num_changes

    @functools.lru_cache(maxsize = None)
    def shortest_time(self, start, end):
        #If x-coords of start point and end point match
        if start[0] == end[0]:
            return sum([self.points[(start[0], coord)] for coord in range(start[1], end[1] + 1)])
        #If y-coords of start point and end point match
        elif start[1] == end[1]:
            return sum([self.points[(coord, start[1])] for coord in range(start[0], end[0] + 1)])
        else:
            return min(self.shortest_time(start, (end[0] - 1, end[1])), self.shortest_time(start, (end[0], end[1] - 1))) + self.points[end]



    @functools.lru_cache(maxsize = None)
    def time_and_path(self, start, end):
        time = self.shortest_time(start, end)
        path = [end]
        current_node = end
        while current_node != start :
            if current_node[0] == start[0]:
                current_node = (current_node[0], current_node[1] - 1)
            elif current_node[1] == start[1]:
                current_node = (current_node[0] - 1, current_node[1])
            elif self.shortest_time(start, (current_node[0] - 1, current_node[1])) >= self.shortest_time(start, (current_node[0], current_node[1] - 1)):
                current_node = (current_node[0], current_node[1] - 1)
            elif self.shortest_time(start, (current_node[0] - 1, current_node[1])) < self.shortest_time(start, (current_node[0], current_node[1] - 1)):
                current_node = (current_node[0] - 1, current_node[1])
            path += [current_node]

        length = len(path)
        real_path = []
        for i in range(length):
            real_path += [path[length - i - 1]]

        return time, real_path


# Part 1.
# Here is the code for the graph & movie.
    def graph_pix(self, filename, time_step, show_weights=False, changed=None):
        plt.clf()
        G = nx.grid_2d_graph(self.m + 1, self.n + 1)
        pos = {}
        for node in G.nodes:
            pos[node] = node
        nx.draw(G, pos, node_size = 0)
        path = self.time_and_path((0, 0), (self.m, self.n))[1]
        path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        if show_weights:
            nx.draw_networkx_labels(G, pos, {point: round(self.points[point], 4) for point in self.points})
            nx.draw_networkx_nodes(G, pos, node_size = 1700, node_color = 'gray', edgecolors='black')
            nx.draw_networkx_edges(G, pos, edge_color='w', alpha=0.5)
            nx.draw_networkx_edges(G, pos, path, 5, edge_color='r')
            if changed:
                nx.draw_networkx_nodes(G, pos, node_size = 1700, nodelist = changed, alpha=0.5, edgecolors='r')
        else:
            nx.draw_networkx_nodes(G, pos, node_size = 0.75, node_color = 'gray', edgecolors='black', alpha=.5)
            nx.draw_networkx_edges(G, pos, width=10, edge_color='w', alpha=0.25)
            nx.draw_networkx_edges(G, pos, path, 1.5, edge_color='r')
            if changed:
                nx.draw_networkx_nodes(G, pos, node_size = 1.5, nodelist = changed, alpha=0.5, edgecolors='r')
        plt.text(40, -4, "Nodes Changed: " + str(time_step))
        plt.savefig("short_path_movie/" + filename + ".png", dpi=200)


    def randmov(self, num_frames, change_num, filename, show_weights=False):
        changed = None
        filenames = []
        for i in range(num_frames):
            self.graph_pix(filename + str(i), i * change_num, show_weights, changed)
            changed = [self.update_lattice() for j in range(change_num)]
            filenames.append("short_path_movie/" + filename + str(i) + ".png")
        images = []
        for file in filenames:
            images.append(imageio.imread(file))
        imageio.mimsave("short_path_movie/" + filename + "_mov.gif", images, duration=0.25)


# Part 2
# The code for difference of two paths while changing the end point.

    @functools.lru_cache(maxsize = None)
    def move_anti_diagonal(self, n):
        k = int(round(math.pow(n, 2/3)))
        start_1 = (2 * k, 0)
        start_2 = (0, 2 * k)
        diff = []
        l = 0
        for i in range(2 * k + 1):
            temp_dist_1 = self.shortest_time(start_1, (n + i, n + 2 * k - i))
            temp_dist_2 = self.shortest_time(start_2, (n + i, n + 2 * k - i))
            diff += [temp_dist_1 - temp_dist_2]
            if i > 0:
                if diff[i] != diff[i - 1]:
                    l += 1
        return diff, l

    def generate_plots(self, n):
        k = int(round(math.pow(n, 2/3)))
        data, l = self.move_anti_diagonal(n)
        x_axis = [i-k for i in range(2 * k + 1)]
        plt.plot(x_axis, data)
        plt.text(40, -4, "Changed Time: " + str(l))
        plt.show()

def generate_multi_points(n, num):
    list_of_l = []
    for i in range(num):
        k = int(round(math.pow(n, 2/3)))
        temp_points = Lattice(n + 2 * k, n + 2 * k)
        temp_l = temp_points.move_anti_diagonal(n)[1]
        list_of_l += [temp_l]
    result = sum(list_of_l) / num
    #print(list_of_l)
    return result

def Plot_for_Alpha(n_min, n_max, num, jump):
    list_of_avg_l = []
    num_of_n = int((n_max - n_min)/jump)
    for i in range(num_of_n):
        temp_n = n_min + jump * i
        temp_avg_l = generate_multi_points(temp_n, num)
        print("n is " + str(temp_n) + " ; with avg_l equals " + str(temp_avg_l))
        list_of_avg_l += [temp_avg_l]
    x_axis = [n_min + jump * i for i in range(num_of_n)]
    plt.plot(x_axis, list_of_avg_l)
    plt.show()
    return list_of_avg_l

def get_time_data(amount, n):
    times = []
    for i in range(amount):
        l = Lattice(n, n)
        times.append(l.time_to_flip())
        print(i)
    return times


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

"""
    @functools.lru_cache(maxsize = None)
    def all_data_horizontal():
    	start = (0, 0)
    	# For 100*100, horizontal to 500*100
    	start_1 = (22, 0)
    	diff_dis_1 = []
    	for i in range(100, 501):
    		temp_dist_1 = shortest_time(start, (i, 100))
    		temp_dist_2 = shortest_time(start_1, (i, 100))
    		diff_dis_1 += [temp_dist_1 - temp_dist_2]

    	# For 500 * 500, horizontal to 1000 * 500
    	start_2 = (63, 0)
    	diff_dis_2 = []
    	for i in range(500, 1001):
    		temp_dist_1 = shortest_time(start, (i, 500))
    		temp_dist_2 = shortest_time(start_2, (i, 500))
    		diff_dis_2 += [temp_dist_1 - temp_dist_2]

    	return diff_dis_1, diff_dis_2

    @functools.lru_cache(maxsize = None)
    def all_data_vertical():
    	start = (0, 0)
    	# For 100*100, horizontal to 500*100
    	start_1 = (22, 0)
    	diff_dis_3 = []
    	for i in range(100, 501):
    		temp_dist_1 = shortest_time(start, (100, i))
    		temp_dist_2 = shortest_time(start_1, (100, i))
    		diff_dis_3 += [temp_dist_1 - temp_dist_2]

    	# For 500 * 500, horizontal to 1000 * 500
    	start_2 = (63, 0)
    	diff_dis_4 = []
    	for i in range(500, 1001):
    		temp_dist_1 = shortest_time(start, (500, i))
    		temp_dist_2 = shortest_time(start_2, (500, i))
    		diff_dis_4 += [temp_dist_1 - temp_dist_2]

    	return diff_dis_3, diff_dis_4

    def first_plot(data_1):
    	x_1 = [i for i in range(100, 501)]
    	plt.plot(x_1, data_1)
    	plt.show()
    def second_plot(data_2):
    	x_2 = [i for i in range(500, 1001)]
    	plt.plot(x_2, data_2)
    	plt.show()
    def third_plot(data_3):
    	x_3 = [i for i in range(100, 501)]
    	plt.plot(x_3, data_3)
    	plt.show()
    def fourth_plot(data_4):
    	x_4 = [i for i in range(500, 1001)]
    	plt.plot(x_4, data_4)
    	plt.show()

    def generate_plots():
        lattice_data = create_lattice(1000, 1000)
        sys.setrecursionlimit(10000)
        data_1, data_2 = all_data_horizontal()
        data_3, data_4 = all_data_vertical()

        fourth_plot(data_4)
"""
