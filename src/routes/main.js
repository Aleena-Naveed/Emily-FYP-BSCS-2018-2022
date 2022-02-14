import React from "react";

import Login from "../pages/login";
import Signup from "../pages/signup";
import Reports from "../pages/reports";
import Profile from "../pages/profile";
import Settings from "../pages/Settings";
import Homepage from "../pages/Homepage";
import Dashboard from "../pages/dashboard";
import Phq from '../pages/phq_session'


import { BrowserRouter as Router, Switch, Route, Redirect, useHistory } from 'react-router-dom';

import { useSelector } from 'react-redux';

export default function Routes() {
    const islogin = useSelector((state) => state.states.islogin)
    const history = useHistory();

    return (
        <Router history={history}>
            {
                islogin ?
                    <Switch>
                        <Route exact path="/">
                            <Dashboard />
                        </Route>
                        <Route path="/reports">
                            <Reports />
                        </Route>
                        
                        <Route path="/settings">
                            <Settings />
                        </Route>
                        <Route path="/profile">
                            <Profile />
                        </Route>
                        <Route path="/session">
                            <Phq />
                        </Route>
                        <Route path="*">
                            <Redirect to='/' />
                        </Route>
                    </Switch>
                    :
                    <Switch>
                        <Route exact path="/">
                            <Homepage />
                        </Route>
                        <Route exact path="/login">
                            <Login />
                        </Route>
                        <Route path="/signup">
                            <Signup />
                        </Route>
                        <Route path="*">
                            <Redirect to="/" />
                        </Route>
                    </Switch>
            }
        </Router>
    )
}