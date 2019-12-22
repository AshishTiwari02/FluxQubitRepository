package com.PageObjectModel;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class RegistrationPage {

	//String baseUrl = "http://newtours.demoaut.com/mercuryregister.php";
	
	WebDriver driver;
	
	By RegLink = By.linkText("REGISTER");
	By FirstName = By.name("firstname");
	By LastName = By.name("lastname");
	By phone = By.name("phone");
	By username = By.name("email");
	By password = By.name("password");
	By confirmpassword = By.name("confirmPassword");
	By registarBtn = By.name("register");
	
	
	//Constructor for Class
	RegistrationPage(WebDriver d)
	{
		driver = d;
		
	}
	public void clickRegLink()
	{
		driver.findElement(RegLink).click();
	}
	public void setFirstName(String fname)
	{
		driver.findElement(FirstName).sendKeys(fname);
	}
	public void setLastName(String lname)
	{
		driver.findElement(LastName).sendKeys(lname);
	}
	
	public void setPhone(String ph)
	{
		driver.findElement(phone).sendKeys(ph);
	}
	public void setUserName(String uname)
	{
		driver.findElement(username).sendKeys(uname);
	}
	public void setPassword(String pwd)
	{
		driver.findElement(password).sendKeys(pwd);
	}
	
	
	public static void main(String[] args) {
		
		

	}

}
