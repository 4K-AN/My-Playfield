package main 

import "fmt"

func add(x,y int) int {
    return x+y
}
func swap(x,y string) (string,string) {
    return y,x

}

func main () {
    fmt.Println(add(10,20))
a,b := swap("AKHMAD","ANAM")
fmt.Println(a,b)

// var a int = 10
}