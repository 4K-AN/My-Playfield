PROCEDURE POP()
  IF count == 0 THEN
    PRINT "Stack Kosong"
  ELSE
    // Simpan item yang akan di-pop (dari index 0)
    item_popped = arr_stack[0]
    PRINT "Item yang di-pop: ", item_popped
    
    // Geser semua elemen (dari index 1 sampai count-1) ke kiri
    FOR i FROM 0 UPTO count - 2
      arr_stack[i] = arr_stack[i+1]
    ENDFOR
    
    // Kurangi jumlah elemen
    count = count - 1
  ENDIF
ENDPROCEDURE