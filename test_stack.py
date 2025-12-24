from stack import Stack

def test_stack():
    # Create a new stack
    stack = Stack()
    
    # Test is_empty when stack is initially created
    assert stack.is_empty() == True
    
    # Test push and peek
    stack.push(5)
    assert stack.peek() == 5
    assert stack.is_empty() == False
    assert stack.size() == 1
    
    # Test multiple pushes
    stack.push(10)
    stack.push(15)
    assert stack.peek() == 15
    assert stack.size() == 3
    
    # Test pop
    top_item = stack.pop()
    assert top_item == 15
    assert stack.size() == 2
    assert stack.peek() == 10
    
    # Test pop until empty
    stack.pop()
    stack.pop()
    assert stack.is_empty() == True
    
    # Test pop on empty stack raises IndexError
    try:
        stack.pop()
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test peek on empty stack raises IndexError
    try:
        stack.peek()
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

# Run the test
test_stack()
print("All stack tests passed successfully!")