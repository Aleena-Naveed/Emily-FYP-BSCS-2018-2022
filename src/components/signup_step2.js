import React from 'react';

import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import { makeStyles } from "@material-ui/core/styles";
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import InputAdornment from "@material-ui/core/InputAdornment";

import sms from "../emily/sms.svg";
import email from "../emily/email.svg";

import { useFormik } from 'formik';
import { validatePin } from '../pages/yup';
import { useSelector, useDispatch } from "react-redux";
import { setuser } from "../pages/stateSlice";

import axios from "axios";
import Snackbar from "@material-ui/core/Snackbar";
import MuiAlert from '@material-ui/lab/Alert';

function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const useStyles = makeStyles({
    inputProps: {
        color: "#7ea2c4",
        fontWeight: "bold",
        marginTop: 20,
    },
    Btn: {
        borderRadius: 15,
        background: "#3585da",
        boxShadow: "3px 3px 6px rgba(0, 0, 0, 0.16)",
        fontWeight: 'bold',
        color: "#fff",
        marginTop: 10,
        marginBottom: 15,
        "&:hover": {
            background: "rgba(53,133,218,0.8)",
        }
    },
    BtnText: {
        fontWeight: 'bold',
        color: "#fff",
        fontSize: 16,
    },
    typographyText: {
        fontWeight: 700,
        textAlign: "center",
    },
    typographyTextSub: {
        fontWeight: 600,
        fontSize: 12,
        textAlign: "center"
    },
    btmText: {
        textAlign: "center"
    },
    startIcon: {
        width: 30,
        height: 30,
    }
})

export default function Step2({ setActiveStep }) {
    const classes = useStyles();
    const user = useSelector((state) => state.states.user);
    const dispatch = useDispatch();
    const [snackbar, setsnackbar] = React.useState({
        open: false,
        msg: "",
        type: ""
    })
    const handleClose = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }

        setsnackbar({ ...snackbar, open: false });
    };

    const formik = useFormik({
        initialValues: {
            smspin: "",
            emailpin: "",
        },
        validationSchema: validatePin,
        onSubmit: (values) => {
            values = { ...values, user: user }
            console.log(values)
            axios.post("http://localhost:5000/users/verify", values)
                .then(res => {
                    if (res.data.success) {
                        setsnackbar({
                            ...snackbar,
                            open: true,
                            msg: "Email and Phone Number Verification Successful",
                            type: "success"
                        })
                        setTimeout(() => {
                            dispatch(setuser(res.data.user));
                            setActiveStep(2);
                        }, 1000)
                    } else {
                        setsnackbar({
                            ...snackbar,
                            open: true,
                            msg: "Incorrect Code",
                            type: "error"
                        })
                    }
                })
                .catch(err => {
                    setsnackbar({
                        ...snackbar,
                        open: true,
                        msg: "Incorrect Code",
                        type: "error"
                    })
                })
        },
    })

    return (
        <Grid>
            <Typography variant="h5" className={classes.typographyText}>VERIFICATION</Typography>
            <Typography
                variant="subtitle2"
                className={classes.typographyTextSub}
            >
                PLEASE ENTER OTP RECIEVED VIA EMAIL AND SMS
            </Typography>
            <form onSubmit={formik.handleSubmit}>
                <TextField
                    fullWidth
                    type="text"
                    name="emailpin"
                    InputLabelProps={{
                        className: classes.inputProps
                    }}
                    inputProps={{
                        className: classes.inputProps,
                    }}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment>
                                <img src={email} alt="Email icon" className={classes.startIcon} />
                            </InputAdornment>
                        )
                    }}
                    value={formik.values.emailpin}
                    onChange={formik.handleChange}
                    error={formik.touched.emailpin && Boolean(formik.errors.emailpin)}
                    helperText={formik.touched.emailpin && formik.errors.emailpin}
                />
                <Grid>
                    <Typography variant="overline" className={classes.typographyText}>
                        RESEND OTP IN 30 SECONDS
                    </Typography>
                </Grid>
                <Button
                    fullWidth
                    variant="contained"
                    type="submit"
                    className={classes.Btn}
                    classes={{
                        contained: classes.Contained,
                        label: classes.BtnText,
                    }}
                >
                    CONTINUE
                </Button>
            </form>
            <Snackbar open={snackbar.open}
                autoHideDuration={6000}
                onClose={handleClose}
            >
                <Alert severity={snackbar.type}>
                    {snackbar.msg}
                </Alert>
            </Snackbar>
        </Grid>
    )
}