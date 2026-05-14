# MODUL 08 API GATEWAY - TEST REPORT

**Tanggal:** _______________
**Nama:** _______________
**NIM:** _______________

---

## 📋 SUMMARY

| Item | Status |
|------|--------|
| GatewayController Created | ☐ Pass ☐ Fail |
| Routes Configured | ☐ Pass ☐ Fail |
| Logging Implemented | ☐ Pass ☐ Fail |
| All 5 Test Cases Pass | ☐ Yes ☐ No |
| Documentation Complete | ☐ Yes ☐ No |

---

## 🧪 TEST CASE RESULTS

### TEST CASE 1: User Login & GET /api/gateway/students

**Objective:** User successfully logs in and accesses student list through gateway

**Steps Performed:**
```
[ ] Step 1: POST /api/login with user credentials
[ ] Step 2: Capture USER_TOKEN from response
[ ] Step 3: GET /api/gateway/students with Bearer token
[ ] Step 4: Verify gateway wrapper in response
```

**Test Result:**
```
Status Expected: 200 OK
Status Received: ________
Response Code: ________
```

**Response Verification:**
- Gateway field present: ☐ Yes ☐ No
- Message field correct: ☐ Yes ☐ No
- Result array contains students: ☐ Yes ☐ No
- Token validation passed: ☐ Yes ☐ No

**Screenshots/Logs:**
```
[Paste Postman response here or attach screenshot]
```

**Verdict:** ☐ PASS ☐ FAIL

**Notes:**
```
_________________________________________________
_________________________________________________
```

---

### TEST CASE 2: User Denied POST /api/gateway/students

**Objective:** User with role "user" cannot create student (403 Forbidden)

**Steps Performed:**
```
[ ] Step 1: Use USER_TOKEN from Test Case 1
[ ] Step 2: POST /api/gateway/students with student data
[ ] Step 3: Verify 403 response
[ ] Step 4: Check error message about role
```

**Test Result:**
```
Status Expected: 403 Forbidden
Status Received: ________
Response Code: ________
```

**Response Verification:**
- Status is 403: ☐ Yes ☐ No
- Message mentions "Access denied": ☐ Yes ☐ No
- Message mentions "role": ☐ Yes ☐ No
- Request blocked by middleware: ☐ Yes ☐ No

**Screenshots/Logs:**
```
[Paste Postman response here or attach screenshot]
```

**Verdict:** ☐ PASS ☐ FAIL

**Notes:**
```
_________________________________________________
_________________________________________________
```

---

### TEST CASE 3: Admin Successfully POST /api/gateway/students

**Objective:** Admin with role "admin" can create student (201 Created)

**Steps Performed:**
```
[ ] Step 1: POST /api/login with admin credentials
[ ] Step 2: Capture ADMIN_TOKEN from response
[ ] Step 3: POST /api/gateway/students with admin token
[ ] Step 4: Verify 201 response and student data
```

**Test Result:**
```
Status Expected: 201 Created
Status Received: ________
Response Code: ________
```

**Response Verification:**
- Status is 201: ☐ Yes ☐ No
- Gateway field present: ☐ Yes ☐ No
- Message indicates forwarded: ☐ Yes ☐ No
- Student data in result: ☐ Yes ☐ No
- NIM matches request: ☐ Yes ☐ No

**Student Created Data:**
```json
NIM: ________
Nama: ________
Mata Kuliah Count: ________
```

**Screenshots/Logs:**
```
[Paste Postman response here or attach screenshot]
```

**Verdict:** ☐ PASS ☐ FAIL

**Notes:**
```
_________________________________________________
_________________________________________________
```

---

### TEST CASE 4: Request Without Token Rejected

**Objective:** Request to gateway without Authorization header is rejected (401)

**Steps Performed:**
```
[ ] Step 1: Prepare GET /api/gateway/students request
[ ] Step 2: REMOVE Authorization header
[ ] Step 3: Send request
[ ] Step 4: Verify 401 response
```

**Test Result:**
```
Status Expected: 401 Unauthorized
Status Received: ________
Response Code: ________
```

**Response Verification:**
- Status is 401: ☐ Yes ☐ No
- Message mentions "Token": ☐ Yes ☐ No
- Request rejected by JWT middleware: ☐ Yes ☐ No
- No data exposed: ☐ Yes ☐ No

**Screenshots/Logs:**
```
[Paste Postman response here or attach screenshot]
```

**Verdict:** ☐ PASS ☐ FAIL

**Notes:**
```
_________________________________________________
_________________________________________________
```

---

### TEST CASE 5: Valid Token But Wrong Role Denied

**Objective:** User token valid, but role insufficient for admin endpoint (403)

**Steps Performed:**
```
[ ] Step 1: Use USER_TOKEN from Test Case 1
[ ] Step 2: DELETE /api/gateway/students/{nim} with user token
[ ] Step 3: Verify 403 response (not 401)
[ ] Step 4: Confirm token was valid but role denied
```

**Test Result:**
```
Status Expected: 403 Forbidden
Status Received: ________
Response Code: ________
```

**Response Verification:**
- Status is 403 (not 401): ☐ Yes ☐ No
- Token was recognized as valid: ☐ Yes ☐ No
- Message mentions "role": ☐ Yes ☐ No
- Role middleware blocked request: ☐ Yes ☐ No

**Screenshots/Logs:**
```
[Paste Postman response here or attach screenshot]
```

**Verdict:** ☐ PASS ☐ FAIL

**Notes:**
```
_________________________________________________
_________________________________________________
```

---

## 📊 OVERALL TEST SUMMARY

| Test Case | Verdict | Comments |
|-----------|---------|----------|
| 1. User GET students | ☐ PASS ☐ FAIL | |
| 2. User blocked POST | ☐ PASS ☐ FAIL | |
| 3. Admin POST success | ☐ PASS ☐ FAIL | |
| 4. No token rejected | ☐ PASS ☐ FAIL | |
| 5. Wrong role denied | ☐ PASS ☐ FAIL | |

**Total Test Cases Passed:** _____ / 5

---

## 🔍 ADDITIONAL TESTING

### Test Profile Gateway
```
Endpoint: GET /api/gateway/profile
Authorization: Bearer {{USER_TOKEN}}
Expected Status: 200
Received Status: ________
Result: ☐ PASS ☐ FAIL
```

### Test Admin Dashboard
```
Endpoint: GET /api/gateway/admin/dashboard
Authorization: Bearer {{ADMIN_TOKEN}}
Expected Status: 200
Received Status: ________
Result: ☐ PASS ☐ FAIL
```

### Test User Dashboard
```
Endpoint: GET /api/gateway/user/dashboard
Authorization: Bearer {{USER_TOKEN}}
Expected Status: 200
Received Status: ________
Result: ☐ PASS ☐ FAIL
```

---

## 📝 LOGGING VERIFICATION

### Log File Location
```
storage/logs/laravel.log
```

### Log Entries Found
```
[ ] GET /gateway/students entry found
[ ] POST /gateway/students entry found
[ ] DELETE /gateway/students/{nim} entry found
[ ] User email captured in logs
[ ] Role captured in logs
```

### Sample Log Entry
```
[Copy actual log entry here]
```

---

## 📖 COMPARISON: DIRECT vs GATEWAY

### Direct Access (/api/students)
- JWT Required: ☐ Yes ☐ No
- Role Check: ☐ Yes ☐ No
- Logging: ☐ Yes ☐ No
- Middleware: ☐ Yes ☐ No
- Security Level: Low ☐ Medium ☐ High

### Gateway Access (/api/gateway/students)
- JWT Required: ☐ Yes ☐ No
- Role Check: ☐ Yes ☐ No
- Logging: ☐ Yes ☐ No
- Middleware: ☐ Yes ☐ No
- Security Level: Low ☐ Medium ☐ High

**Explanation of Differences:**
```
_________________________________________________
_________________________________________________
_________________________________________________
```

---

## 📝 REFLECTION ANSWERS

### Q1: Perbedaan akses langsung vs gateway?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q2: Mengapa API Gateway berguna?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q3: Keuntungan JWT pada gateway?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q4: Mengapa role middleware tetap diperlukan?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q5: Risiko tanpa gateway?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q6: Fungsi logging pada gateway?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q7: Mengapa rate limiting penting?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

### Q8: Laravel vs Kong/Nginx?
```
_________________________________________________
_________________________________________________
_________________________________________________
```

---

## ✅ IMPLEMENTATION CHECKLIST

### Code Implementation
- [x] GatewayController created
- [x] All 7 methods implemented
- [x] Routes added to api.php
- [x] Middleware configured
- [x] Logging added
- [ ] Code reviewed
- [ ] No syntax errors

### Documentation
- [ ] Implementation guide read
- [ ] Testing guide followed
- [ ] Reflection questions answered
- [ ] This report completed

### Testing
- [ ] All 5 test cases passed
- [ ] Additional endpoints tested
- [ ] Logging verified
- [ ] Edge cases tested

---

## 🎯 OVERALL VERDICT

**All Tests Passed:** ☐ Yes ☐ No

**Implementation Quality:**
- Code clarity: ☐ Good ☐ Fair ☐ Needs Improvement
- Documentation: ☐ Complete ☐ Partial ☐ Missing
- Test coverage: ☐ Comprehensive ☐ Adequate ☐ Incomplete

---

## 📝 STUDENT SIGNATURE & DATE

```
Nama: ________________________
NIM:  ________________________
Tanggal: ____________________
Tanda Tangan: _______________
```

---

## 🏫 LECTURER REVIEW

```
Catatan Dosen:
_________________________________________________
_________________________________________________
_________________________________________________

Nilai: _______
Tanda Tangan: _______________
Tanggal: ____________________
```

---

**END OF TEST REPORT**
