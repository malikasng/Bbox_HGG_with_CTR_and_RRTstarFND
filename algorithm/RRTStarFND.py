import random
import math
import numpy as np


class RRT:
    """
    Class for RRT Planning
    """

    def __init__(self, initial_pos, goal_pos, graph, z_dim, real_obs_info, expand_dis=0.07, goal_sample_rate=10, max_iter=100):

        self.start = Node(initial_pos[0], initial_pos[1])
        self.end = Node(goal_pos[0], goal_pos[1])
        # obstacles is list with entries of form: [m_x, m_y, m_z, l, w, h], same format as in mujoco
        self.obstacle_list = graph.obstacles
        # 0 entry means "no obstacle", 1 entry means "obstacle", at the corresponding vertex
        # is_obstacle fct
        self.graph = graph
        self.z_dim = z_dim
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

    def plan(self):
        """
        Path planning
        """
        self.node_list = {0:self.start}
        n = 200
        for i in range(n):
            # get random point of the env
            rnd = self.get_random_point()
            grid_point = None
            # get nearest node index of the graph to random point
            nind = self.get_nearest_list_index(rnd)
            rnd_node = self.steer(rnd, nind)
            # rnd_node: [i, j, k]
            # rnd_node = Node(grid_point)
            # generate new node in direction of random point
            direction_node = self.steer(rnd, nind)

            # find nearest nodes to newNode
            nearinds = self.find_near_nodes(direction_node, 2)
            # from that nearest nodes find the best parent to newNode
            direction_node = self.choose_parent(direction_node, nearinds)

            if not self.is_obstacle(direction_node.x, direction_node.y):  # if it does not collide
                # add direction_node to nodeList
                self.node_list[i + 100] = direction_node
                # make direction_node a parent of another node if necessary
                self.rewire(i + 100, direction_node, nearinds)
                self.node_list[direction_node.parent].children.add(i + 100)
                self.path_validation()

                if len(self.node_list) > self.max_iter:
                    leaves = [key for key, node in self.node_list.items() if
                              len(node.children) == 0 and len(self.node_list[node.parent].children) > 1]
                    if len(leaves) > 1:
                        ind = leaves[random.randint(0, len(leaves) - 1)]
                        self.node_list[self.node_list[ind].parent].children.discard(ind)
                        self.node_list.pop(ind)
                    else:
                        leaves = [key for key, node in self.node_list.items() if len(node.children) == 0]
                        ind = leaves[random.randint(0, len(leaves) - 1)]
                        self.node_list[self.node_list[ind].parent].children.discard(ind)
                        self.node_list.pop(ind)

        best_path_to_goal = self.create_path()
        return best_path_to_goal

    def create_path(self):
        last_index = self.get_best_last_index()
        if last_index is not None:
            path = self.gen_final_course(last_index)
            return path
        else:
            return None

    def path_validation(self):
        last_index = self.get_best_last_index()
        if last_index is not None:
            while self.node_list[last_index].parent is not None:
                node_ind = last_index
                last_index = self.node_list[last_index].parent

                # dx = self.node_list[nodeInd].x - self.node_list[lastIndex].x
                # dy = self.node_list[nodeInd].y - self.node_list[lastIndex].y
                # d = math.sqrt(dx ** 2 + dy ** 2)
                # theta = math.atan2(dy, dx)
                if self.is_obstacle(self.node_list[last_index].x, self.node_list[last_index].y):
                    self.node_list[last_index].children.discard(node_ind)
                    self.remove_branch(node_ind)

    def remove_branch(self, node_ind):
        for ix in self.node_list[node_ind].children:
            self.remove_branch(ix)
        self.node_list.pop(node_ind)

    def choose_parent(self, direction_node, nearinds):
        if len(nearinds) == 0:
            return direction_node

        dlist = []
        for i in nearinds:
            dx = direction_node.x - self.node_list[i].x
            dy = direction_node.y - self.node_list[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            # theta = math.atan2(dy, dx)
            # 0 entry means "no obstacle", 1 entry means "obstacle", at the corresponding vertex
            # False is 0
            if self.is_obstacle(self.node_list[i].x, self.node_list[i].y):
            # if self.check_collision_extend(self.node_list[i].x, self.node_list[i].y, theta, d):
                dlist.append(float("inf"))
            else:
                dlist.append(self.node_list[i].cost + d)

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            return direction_node

        direction_node.cost = mincost
        direction_node.parent = minind
        return direction_node

    def steer(self, rnd, nind):

        nearestNode = self.node_list[nind]
        theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
        newNode = Node(nearestNode.x, nearestNode.y)
        newNode.x += self.expand_dis * math.cos(theta)
        newNode.y += self.expand_dis * math.sin(theta)

        newNode.cost = nearestNode.cost + self.expand_dis
        newNode.parent = nind
        return newNode

    def get_random_point(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(-1, 1),
                   random.uniform(-1, 1)]
        else:  # goal point sampling
            # change end.x to end.get_x_pos_of_a_goal
            rnd = [self.end.x, self.end.y]
        return rnd

    def get_best_last_index(self):

        disglist = [(key, self.calc_dist_to_goal(node.x, node.y)) for key, node in self.node_list.items()]
        goalinds = [key for key, distance in disglist if distance <= self.expand_dis]

        if len(goalinds) == 0:
            return None

        mincost = min([self.node_list[key].cost for key in goalinds])
        for i in goalinds:
            if self.node_list[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [np.array([self.end.x, self.end.y, self.z_dim])]
        while self.node_list[goalind].parent is not None:
            node = self.node_list[goalind]
            path.append(np.array([node.x, node.y, self.z_dim]))
            goalind = node.parent
        path.append(np.array([self.start.x, self.start.y, self.z_dim]))
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, direction_node, value):
        r = self.expand_dis * value
        # check if node[0] works for the initial position
        dlist = np.subtract(np.array([(node.x, node.y) for node in self.node_list.values()]),
                            (direction_node.x, direction_node.y)) ** 2
        dlist = np.sum(dlist, axis=1)
        nearinds = np.where(dlist <= r ** 2)
        # indices (keys) of created nodes
        nearinds = np.array(list(self.node_list.keys()))[nearinds]

        return nearinds

    def rewire(self, direction_node_ind, direction_node, nearinds):
        nnode = len(self.node_list)
        for i in nearinds:
            nearNode = self.node_list[i]

            dx = direction_node.x - nearNode.x
            dy = direction_node.y - nearNode.y
            d = math.sqrt(dx ** 2 + dy ** 2)

            scost = direction_node.cost + d

            if nearNode.cost > scost:
                # theta = math.atan2(dy, dx)
                if not self.is_obstacle(nearNode.x, nearNode.y):
                    self.node_list[nearNode.parent].children.discard(i)
                    nearNode.parent = direction_node_ind
                    nearNode.cost = scost
                    direction_node.children.add(i)

    def get_nearest_list_index(self, rnd):
        dlist = np.subtract( np.array([(node.x, node.y) for node in self.node_list.values() ]), (rnd[0],rnd[1]))**2
        dlist = np.sum(dlist, axis=1)
        minind = list(self.node_list.keys())[np.argmin(dlist)]
        return minind

    # def check_collision_extend(self, nix, niy, theta, d):
    #
    #     tmpNode = Node(nix, niy)
    #
    #     for i in range(int(d / 5)):
    #         tmpNode.x += 5 * math.cos(theta)
    #         tmpNode.y += 5 * math.sin(theta)
    #         if not self.__CollisionCheck(tmpNode, self.obstacleList):
    #             return False
    #
    #     return True

    def is_obstacle(self, node_x, node_y):
        # obstacle_list is a list with entries of form: [max_left, min_right, max_down, min_up]
        for [max_left, min_right, max_down, min_up] in self.obstacle_list:
            if node_x > max_left and node_x < min_right:
                if node_y > max_down and node_y < min_up:
                    return True
        return False


class Node:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.children = set()


# def main():
#     print("start RRT path planning")
#
#     # ====Search Path with RRT====
#     obstacleList = [
#         (400, 380, 400, 20),
#         (400, 220, 20, 180),
#         (500, 280, 150, 20),
#         (0, 500, 100, 20),
#         (500, 450, 20, 150),
#         (400, 100, 20, 80),
#         (100, 100, 100, 20)
#     ]  # [x,y,size]
#     # Set Initial parameters
#     rrt = RRT(start=[20, 580], goal=[540, 150],
#               randArea=[XDIM, YDIM], obstacleList=obstacleList)
#     path = rrt.Planning()
#
#
# if __name__ == '__main__':
#     main()
