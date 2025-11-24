

import java.util.NoSuchElementException;

public class CircularQueue<T> {

    private T[] queue;
    private int front;
    private int rear;
    private int itemCount;
    private int capacity;

    @SuppressWarnings("unchecked")
    public CircularQueue(int initialCapacity) {
        this.capacity = initialCapacity;
        this.queue = (T[]) new Object[capacity];
        this.itemCount = 0;
        this.front = 0;
        this.rear = -1;
    }

    public boolean isEmpty() {
        return itemCount == 0;
    }

    public boolean isFull() {
        return itemCount == capacity;
    }

    public int size() {
        return itemCount;
    }

    @SuppressWarnings("unchecked")
    private void arrayDoubling() {
        int oldCapacity = this.capacity;
        this.capacity *= 2;
        T[] newQueue = (T[]) new Object[this.capacity];

        System.out.println("Kapasitas penuh, melakukan array doubling...");
        for (int i = 0; i < itemCount; i++) {
            newQueue[i] = queue[(front + i) % oldCapacity];
        }

        this.queue = newQueue;
        this.front = 0;
        this.rear = itemCount - 1;
        System.out.println("Array digandakan, kapasitas baru: " + this.capacity);
    }

    public void enqueue(T data) {
        if (isFull()) {
            arrayDoubling();
        }
        rear = (rear + 1) % capacity;
        queue[rear] = data;
        itemCount++;
        System.out.println("Enqueue: " + data);
    }

    public T dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue kosong, tidak bisa dequeue.");
        }
        T data = queue[front];
        queue[front] = null;
        front = (front + 1) % capacity;
        itemCount--;
        System.out.println("Dequeue: " + data);
        return data;
    }

    public void printStatus() {
        System.out.print("   Isi Queue: ");
        for (int i = 0; i < itemCount; i++) {
            System.out.print(queue[(front + i) % capacity] + " ");
        }
        System.out.println("| Size: " + size());
        System.out.println("-------------------------------------");
    }

    public static void main(String[] args) {
        CircularQueue<Integer> q = new CircularQueue<>(3);
        q.printStatus();

        q.enqueue(10);
        q.enqueue(20);
        q.printStatus();

        q.dequeue();
        q.printStatus();

        q.enqueue(30);
        q.enqueue(40);
        q.printStatus();

        q.enqueue(50);
        q.printStatus();
    }
}
