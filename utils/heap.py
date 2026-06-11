class IndexHeap:
    def __init__(self, k=None, order="min", cmp=None):
        self.k = k

        if cmp is not None:
            self.cmp = cmp
        elif order == "min":
            self.cmp = lambda a, b: a < b
        elif order == "max":
            self.cmp = lambda a, b: a > b
        else:
            raise ValueError("order must be 'min' or 'max'")

        self.heap = []              # (priority, idx, value)
        self.pos = {}               # idx -> position

    def __len__(self):
        return len(self.heap)

    def _higher_priority(self, a, b):
        return self.cmp(a, b)

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][1]] = i
        self.pos[self.heap[j][1]] = j

    def _sift_up(self, i):
        while i:
            p = (i - 1) // 2

            if not self._higher_priority(
                self.heap[i][0],
                self.heap[p][0]
            ):
                break

            self._swap(i, p)
            i = p

    def _sift_down(self, i):
        n = len(self.heap)

        while True:
            best = i
            l = 2 * i + 1
            r = l + 1

            if (
                l < n and
                self._higher_priority(
                    self.heap[l][0],
                    self.heap[best][0]
                )
            ):
                best = l

            if (
                r < n and
                self._higher_priority(
                    self.heap[r][0],
                    self.heap[best][0]
                )
            ):
                best = r

            if best == i:
                return

            self._swap(i, best)
            i = best

    def push(self, idx, value, priority=None):
        if priority is None:
            priority = value

        item = (priority, idx, value)

        # still filling top-k
        if self.k is None or len(self.heap) < self.k:
            self.heap.append(item)
            pos = len(self.heap) - 1
            self.pos[idx] = pos
            self._sift_up(pos)
            return True

        # heap full
        root_priority = self.heap[0][0]

        # keep only larger priorities
        if priority <= root_priority:
            return False

        # replace current worst
        old_idx = self.heap[0][1]
        del self.pos[old_idx]

        self.heap[0] = item
        self.pos[idx] = 0
        self._sift_down(0)

        return True
        
    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty heap")

        priority, idx, value = self.heap[0]

        last = self.heap.pop()
        del self.pos[idx]

        if self.heap:
            self.heap[0] = last
            self.pos[last[1]] = 0
            self._sift_down(0)

        return value

    def at(self, idx):
        return self.heap[self.pos[idx]][2]

    def peek(self):
        if not self.heap:
            raise IndexError("empty heap")
        return self.heap[0][2]

    def priority(self, idx):
        return self.heap[self.pos[idx]][0]

    def contains(self, idx):
        return idx in self.pos