# Leetcode Cheatsheet 

## Sorting
Best used for problems like interval merging
Stick to Python builtin
```python
bla.sort()
```

but also good to memorize one sorting algorithm just in case. Why not Quicksort
Runtime $O(n \log n)$ on average Worstcase $O(n^2)$
```Python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2] # random.choice(arr) is better for average runtime
   
    less  = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    
    return quicksort(less) + equal + quicksort(greater)
```

## Quickselect
Simply abbreviated from Quicksort we can define the quickselect algorithm. Good for e.g.: Find the kth smallest element in an unordered list
Average runtime is $O(n)$ since we only need to traverse into the "correct" side. Worst-case is still $O(n^2)$
```Python
def quickselect(arr: List[T], k: int) -> T:
    if not 0 <= k < len(arr):
        raise IndexError("k is out of bounds")

    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr)

    lows  = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    if k < len(lows):
        return quickselect(lows, k)             # in the lows
    elif k < len(lows) + len(pivots):
        return pivots[0]                        # it's equal to pivot
    else:
        # skip lows & pivots
        return quickselect(highs, k - len(lows) - len(pivots))
```


## Binary Search

Find something in an ordered list in $O(\log n)$ time. The classic recursive algorithm is:

```Python
def binary_search(arr, target, left, right):
    if left > right:
        return -1  # base case: not found

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)
```

iterative is simply a two pointer algorithm:
```python
def binary_search_iterative(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

Most leetcode medium questions require to modify the binary search algorithm so go with the iterative one.

#### Example: Search in Rotated Sorted Array

You are given an integer array nums sorted in ascending order (with distinct values), which is rotated at an unknown pivot index k (e.g., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`).

```Python
 left, right = 0, len(nums) - 1
    
while left <= right:
    mid = (left + right) // 2
    
    if nums[mid] == target:
        return mid
    
    # Left half is sorted
    if nums[left] <= nums[mid]:
        if nums[left] <= target < nums[mid]:
            right = mid - 1  # search in left
        else:
            left = mid + 1   # search in right
    else:
        # Right half is sorted
        if nums[mid] < target <= nums[right]:
            left = mid + 1  # search in right
        else:
            right = mid - 1  # search in left

return -1
```

#### Example: Leetcode 1060
```Given an integer array nums which is sorted in ascending order and all of its elements are unique and given also an integer k, return the kth missing number starting from the leftmost number of the array.```

We first check if k is larger than the number of missing elements from i=0 to i=n - 1
Then we use a binary search to split the list in half
 - left side has less missing elements than k. then we move the window one to the right
 - left side has more elements than k then we know we already overshoot

at the end we know that left points to the element before we need to insert the missind value
```python
n = len(nums)
def missing(index: int) -> int:
    return nums[index] - nums[0] - index

if k > missing(n - 1):
    return nums[-1] + (k - missing(n - 1))

left, right = 0, n - 1
while left < right:
    mid = (left+right)//2

    if missing(mid) < k:
        left += 1
    else:
        right = mid

return nums[left - 1] + (k - missing(left - 1))
```

For plain old binary search we can also use Pythons builtin `bisect` , `bisect_right` or `bisect_left`.

#### Find First True in a monotonic boolean list
```Python
bools = [False, False, False, True, True]
print(bisect_left(bools, True)) # --> 3
```
### Find First element ≥ target
```Python
arr = [1, 2, 4, 4, 5, 7, 8]
print(bisect_left(arr, 4)) # --> 2
```
### Find First element > target
```Python
print(bisect_right(arr, 4)) # --> 4
```

### Find Last element ≤ target
```Python
print(bisect_right(arr, 3) - 1) # --> 3 
```

### Find Rightmost element == target
```Python
lo = bisect_left(arr, 4)
hi = bisect_right(arr, 4)
print(hi - 1 if lo < hi else None) # --> 3

```

## Unionfind and Connected Components

These are one of my favourite algorithms. It is required for LC mediums as in `Leetcode 721: Accounts Merge`.
`Given a list of accounts where each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account. Now, we would like to merge these accounts.`

### Connected Components

Connected components returns a list of lists containing components that are literally connected.
The runtime and space complexity is $O(n + e)$ .
We need to build an adjacency list. The algorithm traverse each unseen node and does a DFS collecting neighbors.
```Python
def connected_components(neighbors):
    seen = set()
    def component(node):
        nodes = set([node])
        res = []
        while nodes:
            node = nodes.pop()
            seen.add(node)
            for neighbor in neighbors[node]:
                if neighbor not in seen:
                    nodes.add(neighbor)
            res.append(node)
        return res

    ret = []
    for node in neighbors:
        if node not in seen:
            ret.append(component(node))
    return ret
```

#### Example: Solving Accounts Merge
We draw an edge from a random first email to any other email to create the connected component. Then we calculate them using the algorithms.
and finally just map them back to the name of the account holder and sort them as required by the task.

```Python
graph = defaultdict(set)
email_to_name = {}
for account in accounts:
    name = account[0]
    first_email = account[1]
    for email in account[1:]:
        graph[first_email].add(email)
        graph[email].add(first_email)
        email_to_name[email] = name

components = connected_components(graph)

new_accounts = []
for acc in components:
    name = email_to_name[acc[0]]
    mails = sorted(acc)
    new_accounts.append([name] + mails)
return new_accounts
```

### Union Find

The union find algorithm is a bit more intriguing. For each entry we want to find a greatest common parent.
A lookup is $O(\alpha(n))$ which is the inverse Ackerman function. We can treat it as constant for $n < 10^7$.
It has $O(n)$ space complexity for storing all nodes.
```python
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # Path compression
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)
```

#### Example: Solving Accounts Merge
Instead of drawing an edge from email to email we union them by their common root.
```Python
for account in accounts:
        name = account[0]
        emails = account[1:]
        for email in emails:
            if email not in parent:
                parent[email] = email
            email_to_name[email] = name
        for i in range(1, len(emails)):
            union(emails[0], emails[i])

    # Step 2: Group emails by root parent
groups = defaultdict(list)
for email in parent:
    root = find(email)
    groups[root].append(email)
```

## Backtracking

Backtracking explores the problem space and eliminates candidates that do not fit a certain criterion.
IMO its mostly about using a DFS or BFS and keeping track of options.
```Python
def backtrack(path, options):
    if base_case_condition:
        res.append(path[:])  # save a copy of the solution
        return

    for i in range(len(options)):
        # Choose
        path.append(options[i])

        # Explore
        backtrack(path, updated_options)

        # Un-choose (backtrack)
        path.pop()
```

### Example: Leetcode 46: Permutations (Medium)

Given an array nums of distinct integers, return all the possible permutations.
The backtracking solution has $O(n!)$ runtime complexity.
```Python
def permute(nums):
    res = []
    stack = [([], nums)]

    while stack:
        path, options = stack.pop()

        if not options:
            res.append(path)
            continue

        for i in range(len(options)):
            new_path = path + [options[i]]
            new_options = options[:i] + options[i+1:]
            stack.append((new_path, new_options))

    return res
```

### Example: Leetcode 22

All combinations of well-formed parentheses given $n$ pairs.
```Python
res = []

def backtrack(current: str, open_count: int, close_count: int):
    if len(current) == 2 * n:
        res.append(current)
        return
    if open_count < n:
        backtrack(current + '(', open_count + 1, close_count)
    if close_count < open_count:
        backtrack(current + ')', open_count, close_count + 1)

backtrack("", 0, 0)
return res
```

Funnily the algorithm has a Catalan number runtime complexity of $O(4^n / \sqrt n)$ .

## Heap
Get the max or min value of a heap in $O(1)$ and Pushing/Poping in $O(\log n)$ 
In python we can simply use:

```python
heap = []
heappush(heap, (0, "Hello"))
r = heappop(heap)
```
Or also heapifying a list inplace in linear time:
```python
heapify(list)
```
In Python the heap is always a min-heap. You will always pop the min value. To get a max heap simply push the negative values.
Also neat is that you can push anything to this DS as long as that it will sort on the first index of your element.

### Example: Leetcode 502

You are given two arrays:
- `profits[i]` — profit of the ith project 
- `capital[i]` — minimum capital required to start the ith project
and two integers:
- $k$ — max number of projects you can do
- $w$ — initial capital

Return the maximum capital you can have after completing at most k projects.

This is a LC hard question. But the solution is:
1. Sort projects by required capital
2. Use a max-heap to store profits of projects you can currently afford (capital[i] ≤ w)
3. At each step:
   1. Add all newly affordable projects to the heap 
   2. Pop the most profitable one 
   3. Add its profit to your capital w 
4. Repeat up to k times



```python
projects = list(zip(capital, profits))
projects.sort()  # sort by capital required

max_heap = []
i = 0

for _ in range(k):
    # Push all projects we can afford into max-heap
    while i < len(projects) and projects[i][0] <= w:
        heapq.heappush(max_heap, -projects[i][1])  # use neg profit for max-heap
        i += 1

    if not max_heap:
        break

    w += -heapq.heappop(max_heap)  # take most profitable available

return w
```

## Stack

Well. Its a stack.

### Example: 38 Count and Say


The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

countAndSay(1) = "1"
countAndSay(n) is the run-length encoding of countAndSay(n - 1).

For isntance:
Input: $n = 4$
Output: "1211"

```Python
stack = ["1"]  # Start with the base case for n = 1
c = 1
while c < n:
    s = stack.pop()
    encoded = ""
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            encoded += str(count) + s[i - 1]
            count = 1

    encoded += str(count) + s[-1]
    stack.append(encoded)
    c += 1

return stack.pop()
```
For the runtime complexity we first see that
$|f(s + t)| ≤ |f(s)| + |f(t)|$ since a merge will at most reduce the overall length. This is called Subadditivity.
Now if we compress any digit other than one it will at most become fourfold:
- f("d") → "1d"
- f("1d") → "111d"
- f("111d") → "311d"

Roughly after 3 iterations. So we can estimate an loose upper bound for the runtime complexity of $O(4^\frac{n}{3})$


## Special Algorithms

### Moores Voting

Given a list. Assuming there is a majority of some element. E.g. ["yes", "yes", "no"]. We can find the most frequent element with $O(1)$ space and $O(n)$ runtime.
```Python
winner = l[0]
counter = 0
for n in l:
    if winner == n:
        counter += 1
    elif:
        counter -= 1
    if counter == 0:
        winner = n
```

### Kadane algorithm

Find a non-empty subarray with the largest sum. In $O(n)$ runtime and $O(1)$ Space,
```Python
def kadanes(nums):
    maxSum = nums[0]
    curSum = 0

    for n in nums:
        curSum = max(curSum, 0)
        curSum += n
        maxSum = max(maxSum, curSum)
    return maxSum
```

## Prefix Sum

The key idea is: `prefix_sum[i] = sum of nums[0..i]`


### Example:  Prefix Sum – Subarray Sum Equals K

Given an array of integers nums and an integer $k$, return the total number of subarrays whose sum equals to $k$.
REMEMBER: subarray is a contiguous slice of the original array

The key idea of the algorithms is:
`prefix_sum[j] - prefix_sum[i - 1] == k`

and if rearranged a bit: `prefix_sum[i - 1] == prefix_sum[j] - k`

- Input: nums = [1,1,1], k = 2
- Output: 2

```Python
def subarraySum(nums, k):
    prefix_sum = 0
    count = 0
    freq = defaultdict(int)
    freq[0] = 1  # base case
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in freq:
            count += freq[prefix_sum - k]
        freq[prefix_sum] += 1
    
    return count
    
```

$O(n)$ time and $O(n)$ space complexity.
Why is this Hashmap + PrefixSum approach working?
PrefixSum is monotonically increasing.