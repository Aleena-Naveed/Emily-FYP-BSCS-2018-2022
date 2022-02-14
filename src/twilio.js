var express=require ("express")
const app = express()

const port = 5000
const config = require("./config")
const client = require ("twilio")

// /login 
//     phonenumber
//     channel(call/sms)

var sid = new client("AC5909287c9abb7201f8512b6ab7d3a0a0", "3a9db5b183f2d38bcb0aaf022917cbb1")
var contact = new contact;
var code = new code;


app.get('/login', (req,res)=>{
    sid
        .verify
        .services("VA4fe7f5eff304278cb36d18d37ae2669a")
        .verifications
        .create({
            to: {contact},
            channel: "sms"
        })
        .then((data)=>{
            res.status(200).send(data)
        })
})

app.get('/verify', (req,res)=>{
    sid
        .verify
        .services("VA4fe7f5eff304278cb36d18d37ae2669a")
        .verificationChecks
        .create({
            to: { contact },
            code: {code} 
        })
        .then((data)=>{
            res.status(200).send(data)
        })
})

app.listen(port, () => {
    console.log("Server is running at ${port}" )
})