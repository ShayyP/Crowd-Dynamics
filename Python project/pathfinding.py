# !!!Deprecated file!!!

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class AStar:
    def __init__(self, grid_size, exit_pos):
        self._grid_size = grid_size
        self._exit_pos = exit_pos
        self._closed_list = []

    def _is_valid(self, pos):
        return (pos[0] >= 0) and (pos[0] < self._grid_size[0]) and (pos[1] >= 0) and (pos[1] < self._grid_size[1])

    def _calculate_h_value(self, pos):
        return ((pos[0] - self._exit_pos[0]) ** 2) + ((pos[1] - self._exit_pos[1]) ** 2)

    @staticmethod
    def _sort_list(lst):
        lst.sort(key=lambda x: x.f)

    @staticmethod
    def _get_path(current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1][1::]

    def search(self, grid, src):
        if not (self._is_valid(src) and self._is_valid(self._exit_pos)):
            print("Source or destination is invalid")
            return False

        start_node = Node(None, src)
        end_node = Node(None, self._exit_pos)

        open_list = [start_node]
        closed_list = []

        while len(open_list) != 0:
            self._sort_list(open_list)
            current = open_list[0]
            open_list.remove(current)
            closed_list.append(current)

            i, j = current.position
            neighbours = [(i, j-1), (i, j+1), (i+1, j), (i-1, j), (i+1, j-1), (i-1, j-1), (i+1, j+1), (i-1, j+1)]
            children = []
            for next_i, next_j in neighbours:
                if not self._is_valid((next_i, next_j)):
                    continue

                if (next_i, next_j) == self._exit_pos:
                    end_node.parent = current
                    return self._get_path(end_node)

                elif grid[next_j][next_i].is_wall():
                    continue

                children.append(Node(current, (next_i, next_j)))

            for child in children:
                if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                    continue

                child.g = current.g + 1
                child.h = self._calculate_h_value(child.position)
                child.f = child.g + child.h

                if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                    continue

                open_list.append(child)

        print("No path found")
        return False
