import java.util.*;

class Node {
    int data;
    Node left, right;
    int height;
    
    public Node(int data) {
        this.data = data;
        this.left = null;
        this.right = null;
        this.height = 1;
    }
}

class AVLTree {
    private Node root;
    private int count;
    private int capacity;
    
    public AVLTree(int capacity) {
        this.root = null;
        this.count = 0;
        this.capacity = capacity;
    }
    
    private int height(Node node) {
        if (node == null) return 0;
        return node.height;
    }

    private int getBalance(Node node) {
        if (node == null) return 0;
        return height(node.left) - height(node.right);
    }
    

    private void updateHeight(Node node) {
        if (node != null) {
            node.height = Math.max(height(node.left), height(node.right)) + 1;
        }
    }
    
 
    private Node rightRotate(Node y) {
        Node x = y.left;
        Node T2 = x.right;
        
      
        x.right = y;
        y.left = T2;
        
     
        updateHeight(y);
        updateHeight(x);
        
        return x;
    }
    

    private Node leftRotate(Node x) {
        Node y = x.right;
        Node T2 = y.left;
        
      
        y.left = x;
        x.right = T2;
        
  
        updateHeight(x);
        updateHeight(y);
        
        return y;
    }
    
  
    private boolean search(Node node, int data) {
        if (node == null) return false;
        
        if (node.data == data) return true;
        
        if (data < node.data) {
            return search(node.left, data);
        } else {
            return search(node.right, data);
        }
    }
    
 
    public void insert(int data) {
        if (count >= capacity) {
            System.out.println("Error: tree sudah penuh");
            return;
        }
        
        if (search(root, data)) {
            System.out.println("Error: data sudah ada dalam tree");
        } else {
            root = insertRec(root, data);
            count++;
            System.out.println(data + " dimasukkan ke dalam tree");
        }
    }
    
    private Node insertRec(Node node, int data) {
       
        if (node == null) {
            return new Node(data);
        }
        
        if (data < node.data) {
            node.left = insertRec(node.left, data);
        } else if (data > node.data) {
            node.right = insertRec(node.right, data);
        } else {
            return node; 
        }
        
     
        updateHeight(node);
        
      
        int balance = getBalance(node);
        
   
        if (balance > 1 && data < node.left.data) {
            return rightRotate(node);
        }
        
      
        if (balance < -1 && data > node.right.data) {
            return leftRotate(node);
        }
        
    
        if (balance > 1 && data > node.left.data) {
            node.left = leftRotate(node.left);
            return rightRotate(node);
        }
        
 
        if (balance < -1 && data < node.right.data) {
            node.right = rightRotate(node.right);
            return leftRotate(node);
        }
        
        return node;
    }
    

    public void inorder() {
        if (root == null) {
            System.out.println("Error: tree kosong");
        } else {
            System.out.print("Inorder: ");
            inorderRec(root);
            System.out.println();
        }
    }
    
    private void inorderRec(Node node) {
        if (node != null) {
            inorderRec(node.left);
            System.out.print(node.data + " ");
            inorderRec(node.right);
        }
    }
    

    public void countNodes() {
        System.out.println("Jumlah node: " + count);
    }
    
  
    public void treeHeight() {
        int h = height(root);
        System.out.println("Tinggi tree: " + h);
    }
    
 
    public void clear() {
        root = null;
        count = 0;
        System.out.println("Tree dikosongkan");
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
    
        if (!sc.hasNextInt()) {
            System.out.println("Error: kapasitas tree harus antara 1 hingga 100");
            sc.close();
            return;
        }
        
        int n = sc.nextInt();
        sc.nextLine(); 
        
     
        if (n < 1 || n > 100) {
            System.out.println("Error: kapasitas tree harus antara 1 hingga 100");
            sc.close();
            return;
        }
        
        AVLTree tree = new AVLTree(n);
        
   
        while (sc.hasNextLine()) {
            String line = sc.nextLine().trim();
            if (line.isEmpty()) continue;
            
            String[] parts = line.split("\\s+", 2);
            String command = parts[0].toUpperCase();
            
            if (command.equals("INSERT")) {
                if (parts.length < 2 || parts[1].trim().isEmpty()) {
                    System.out.println("Error: nilai tidak boleh kosong atau bukan angka");
                } else {
                    try {
                        int nilai = Integer.parseInt(parts[1].trim());
                        tree.insert(nilai);
                    } catch (NumberFormatException e) {
                        System.out.println("Error: nilai tidak boleh kosong atau bukan angka");
                    }
                }
            } else if (command.equals("INORDER")) {
                tree.inorder();
            } else if (command.equals("COUNT")) {
                tree.countNodes();
            } else if (command.equals("HEIGHT")) {
                tree.treeHeight();
            } else if (command.equals("CLEAR")) {
                tree.clear();
            } else {
                System.out.println("Error: perintah tidak dikenal");
            }
        }
        
        sc.close();
    }
}