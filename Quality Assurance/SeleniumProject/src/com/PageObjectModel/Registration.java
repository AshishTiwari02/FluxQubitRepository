package com.PageObjectModel;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxProfile;

public class Registration {

	public void verifyReg()
	{
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";
		
		System.setProperty(driverPath, driverURL);
		WebDriver driver = new ChromeDriver();
		
		RegistrationPage rPage = new RegistrationPage(driver);
		rPage.clickRegLink();
		rPage.setFirstName("Ashish");
		rPage.setLastName("Tiwari");
		rPage.setPhone("123456");
		
	}
	
	public void verifyRegPage2()
	{
		String driverPath = "webdriver.gecko.driver";
	    String driverURL = "D:\\Softwares\\Selenium\\geckodriver-v0.26.0-win64\\geckodriver.exe"; 
	    
	    System.setProperty(driverPath,driverURL);
	    WebDriver driver  = new FirefoxDriver();
		RegistrationPage2 rPage2 = new RegistrationPage2(driver);
		
		rPage2.setFirstName("Ashish");
		rPage2.setLastName("Tiwari");
		
	}
	public static void main(String[] args) {
		

	}

}
