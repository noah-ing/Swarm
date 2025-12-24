class Stack:
    """
    A simple stack data structure with basic operations.
    
    Attributes:
        _items (list): Internal list to store stack elements.
    """
    
    def __init__(self):
        """
        Initialize an empty stack.
        """
        self._items = []
    
    def push(self, item):
        """
        Add an item to the top of the stack.
        
        Args:
            item: The element to be added to the stack.
        """
        self._items.append(item)
    
    def pop(self):
        """
        Remove and return the top item from the stack.
        
        Returns:
            The top item of the stack.
        
        Raises:
            IndexError: If the stack is empty.
        """
        if not self._items:
            raise IndexError("Cannot pop from an empty stack")
        return self._items.pop()
    
    def peek(self):
        """
        Return the top item of the stack without removing it.
        
        Returns:
            The top item of the stack.
        
        Raises:
            IndexError: If the stack is empty.
        """
        if not self._items:
            raise IndexError("Cannot peek an empty stack")
        return self._items[-1]
    
    def is_empty(self):
        """
        Check if the stack is empty.
        
        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return len(self._items) == 0
    
    def size(self):
        """
        Return the number of items in the stack.
        
        Returns:
            int: Number of items in the stack.
        """
        return len(self._items)