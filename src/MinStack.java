import java.util.Stack;

class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> Minstack;


    public MinStack() {
        stack = new Stack<>();
        Minstack = new Stack<>();
    }

    public void push(int val) {
        stack.push(val);
        if (Minstack.isEmpty()||Minstack.peek()>val){
            Minstack.push(val);
        }
    }

    public void pop() {
        if (stack.pop()==Minstack.peek()){
            Minstack.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return Minstack.peek();
    }
}