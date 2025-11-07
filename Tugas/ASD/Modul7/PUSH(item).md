PROCEDURE PUSH(item)
  IF count == MAX_SIZE THEN
    PRINT "Stack Penuh"
  ELSE
    // Geser semua elemen (dari index count-1 sampai 0) ke kanan
    FOR i FROM count DOWNTO 1
      arr_stack[i] = arr_stack[i-1]
    ENDFOR
    
    // Masukkan item baru di index 0
    arr_stack[0] = item
    // Tambah jumlah elemen
    count = count + 1
  ENDIF
ENDPROCEDURE