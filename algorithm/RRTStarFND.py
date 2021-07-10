import random
import math
import numpy as np

from envs.distance_graph import DistanceGraph


class RRT:
    """
    Class for RRT Planning
    """

    def __init__(self, initial_pos, goal_pos, graph: DistanceGraph, expandDis=1.0, goalSampleRate=10):

        self.start = initial_pos
        self.end = goal_pos

        # obstacles is list with entries of form: [m_x, m_y, m_z, l, w, h], same format as in mujoco
        self.obstacle_list = graph.obstacles
        # 0 entry means "no obstacle", 1 entry means "obstacle", at the corresponding vertex
        # is_obstacle fct
        self.graph = graph
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate

    def plan_paths(self):
        """
        Path planning
        """
        self.created_nodes = {0:self.start}
        n = 50
        for i in range(n):
            # get random point of the env
            rnd = self.get_random_point()
            grid_point = None
            # get nearest node index of the graph to random point
            while grid_point is None:
                grid_point = self.graph.coords2gridpoint(rnd)
            # rnd_node: [i, j, k]
            rnd_node = Node(grid_point)
            # generate new node in direction of random point
            direction_node = self.steer(rnd, rnd_node)

            nearinds = self.find_near_nodes(direction_node, 2)  # find nearest nodes to newNode
            newNode = self.choose_parent(direction_node,
                                         nearinds)  # from that nearest nodes find the best parent to newNode
            self.nodeList[i + 100] = newNode  # add newNode to nodeList
            self.rewire(i + 100, newNode, nearinds)  # make newNode a parent of another node if necessary
            self.nodeList[newNode.parent].children.add(i + 100)

            if len(self.nodeList) > self.maxIter:
                leaves = [key for key, node in self.nodeList.items() if
                          len(node.children) == 0 and len(self.nodeList[node.parent].children) > 1]
                if len(leaves) > 1:
                    ind = leaves[random.randint(0, len(leaves) - 1)]
                    self.nodeList[self.nodeList[ind].parent].children.discard(ind)
                    self.nodeList.pop(ind)
                else:
                    leaves = [key for key, node in self.nodeList.items() if len(node.children) == 0]
                    ind = leaves[random.randint(0, len(leaves) - 1)]
                    self.nodeList[self.nodeList[ind].parent].children.discard(ind)
                    self.nodeList.pop(ind)

    def path_validation(self):
        lastIndex = self.get_best_last_index()
        if lastIndex is not None:
            while self.nodeList[lastIndex].parent is not None:
                nodeInd = lastIndex
                lastIndex = self.nodeList[lastIndex].parent

                dx = self.nodeList[nodeInd].x - self.nodeList[lastIndex].x
                dy = self.nodeList[nodeInd].y - self.nodeList[lastIndex].y
                d = math.sqrt(dx ** 2 + dy ** 2)
                theta = math.atan2(dy, dx)
                if not self.check_collision_extend(self.nodeList[lastIndex].x, self.nodeList[lastIndex].y, theta, d):
                    self.nodeList[lastIndex].children.discard(nodeInd)
                    self.remove_branch(nodeInd)

    def remove_branch(self, nodeInd):
        for ix in self.nodeList[nodeInd].children:
            self.remove_branch(ix)
        self.nodeList.pop(nodeInd)

    def choose_parent(self, newNode, nearinds):
        if len(nearinds) == 0:
            return newNode

        dlist = []
        for i in nearinds:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            if self.check_collision_extend(self.nodeList[i].x, self.nodeList[i].y, theta, d):
                dlist.append(self.nodeList[i].cost + d)
            else:
                dlist.append(float("inf"))

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode.cost = mincost
        newNode.parent = minind
        return newNode

    def steer(self, rnd, rnd_node):

        # expand tree
        theta = math.atan2(rnd[1] - rnd_node.hgg_node[1], rnd[0] - rnd_node.hgg_node[0])
        direction_node = rnd_node
        direction_node.hgg_node[0] += self.expandDis * math.cos(theta)
        direction_node.hgg_node[1] += self.expandDis * math.sin(theta)
        direction_node.hgg_node = self.graph.coords2gridpoint(direction_node.hgg_node)

        direction_node.cost = rnd_node.cost + self.expandDis
        direction_node.parent = rnd_node
        return direction_node

    def get_random_point(self):
        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.graph.x_min, self.graph.x_max),
                   random.uniform(self.graph.y_min, self.graph.y_max),
                   random.uniform(self.graph.z_min, self.graph.z_max)]
        else:  # goal point sampling
            # change end.x to end.get_x_pos_of_a_goal
            rnd = [self.end.x, self.end.y, self.end.z]
        return rnd

    def get_best_last_index(self):

        disglist = [(key, self.calc_dist_to_goal(node.x, node.y)) for key, node in self.nodeList.items()]
        goalinds = [key for key, distance in disglist if distance <= self.expandDis]

        if len(goalinds) == 0:
            return None

        mincost = min([self.nodeList[key].cost for key in goalinds])
        for i in goalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append([node.x, node.y])
            goalind = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, direction_node, value):
        r = self.expandDis * value
        # check if node[0] works for the initial position
        dlist = np.subtract(np.array([(node[0], node[1]) for node in self.created_nodes.values()]),
                            (direction_node.hgg_node[0], direction_node.hgg_node[1])) ** 2
        dlist = np.sum(dlist, axis=1)
        nearinds = np.where(dlist <= r ** 2)
        # indices (keys) of created nodes
        nearinds = np.array(list(self.created_nodes.keys()))[nearinds]

        return nearinds

    def rewire(self, newNodeInd, newNode, nearinds):
        nnode = len(self.nodeList)
        for i in nearinds:
            nearNode = self.nodeList[i]

            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            d = math.sqrt(dx ** 2 + dy ** 2)

            scost = newNode.cost + d

            if nearNode.cost > scost:
                theta = math.atan2(dy, dx)
                if self.check_collision_extend(nearNode.x, nearNode.y, theta, d):
                    self.nodeList[nearNode.parent].children.discard(i)
                    nearNode.parent = newNodeInd
                    nearNode.cost = scost
                    newNode.children.add(i)

    def check_collision_extend(self, nix, niy, theta, d):

        tmpNode = Node(nix, niy)

        for i in range(int(d / 5)):
            tmpNode.x += 5 * math.cos(theta)
            tmpNode.y += 5 * math.sin(theta)
            if not self.__CollisionCheck(tmpNode, self.obstacleList):
                return False

        return True

    # def collision_check(self, node):
    #     # obstacles is list with entries of form: [m_x, m_y, m_z, l, w, h], same format as in mujoco
    #     for [m_x, m_y, m_z, l, w, h] in self.obstacle_list:
    #         m_x, m_y, m_z, l, w, h = sx + 2, sy + 2, ex + 2, ey + 2
    #         if node.x > sx and node.x < sx + ex:
    #             if node.y > sy and node.y < sy + ey:
    #                 return False
    #
    #     return True


class Node:
    """
    RRT Node
    """

    def __init__(self, hgg_node):
        self.hgg_node = hgg_node
        self.cost = 0.0
        self.parent = None
        self.children = set()


def main():
    print("start RRT path planning")

    # ====Search Path with RRT====
    obstacleList = [
        (400, 380, 400, 20),
        (400, 220, 20, 180),
        (500, 280, 150, 20),
        (0, 500, 100, 20),
        (500, 450, 20, 150),
        (400, 100, 20, 80),
        (100, 100, 100, 20)
    ]  # [x,y,size]
    # Set Initial parameters
    rrt = RRT(start=[20, 580], goal=[540, 150],
              randArea=[XDIM, YDIM], obstacleList=obstacleList)
    path = rrt.Planning()


if __name__ == '__main__':
    main()
