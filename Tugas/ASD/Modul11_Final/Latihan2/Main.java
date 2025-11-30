import java.util.*;

class Node {
    int noRak;
    int harga;
    Node left, right;
    
    Node(int noRak, int harga) {
        this.noRak = noRak;
        this.harga = harga;
        this.left = null;
        this.right = null;
    }
}

class BST {
    Node root;
    
    BST() {
        this.root = null;
    }
    
    void input(int noRak, int harga) {
        root = insertRec(root, noRak, harga);
    }
    
    Node insertRec(Node node, int noRak, int harga) {
        if (node == null) {
            return new Node(noRak, harga);
        }
        
        if (noRak < node.noRak) {
            node.left = insertRec(node.left, noRak, harga);
        } else if (noRak > node.noRak) {
            node.right = insertRec(node.right, noRak, harga);
        } else {
            node.harga = harga;
        }
        
        return node;
    }
    
    void cek(int noRak) {
        Integer harga = searchRec(root, noRak);
        if (harga != null) {
            System.out.println("Harga: " + harga);
        } else {
            System.out.println("Harga tidak ditemukan");
        }
    }
    
    Integer searchRec(Node node, int noRak) {
        if (node == null) {
            return null;
        }
        
        if (noRak == node.noRak) {
            return node.harga;
        }
        
        if (noRak < node.noRak) {
            return searchRec(node.left, noRak);
        } else {
            return searchRec(node.right, noRak);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        sc.nextLine();
        
        BST bst = new BST();
        
        for (int i = 0; i < n; i++) {
            String line = sc.nextLine().trim();
            String[] parts = line.split("\\s+");
            String cmd = parts[0].toUpperCase();
            
            if (cmd.equals("INPUT")) {
                if (parts.length >= 3) {
                    int noRak = Integer.parseInt(parts[1]);
                    int harga = Integer.parseInt(parts[2]);
                    bst.input(noRak, harga);
                }
            } else if (cmd.equals("CEK")) {
                if (parts.length >= 2) {
                    int noRak = Integer.parseInt(parts[1]);
                    bst.cek(noRak);
                }
            }
        }
        
        sc.close();
    }
}